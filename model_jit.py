# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, rope_q=None, rope_k=None):
        B, N, C = x.shape
        _, M, _ = context.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_q is not None:
            q = rope_q(q)
        if rope_k is not None:
            k = rope_k(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                                         attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm3 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, cond=None, feat_rope=None, cond_rope=None):
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        if cond is not None:
            x = x + gate_cross.unsqueeze(1) * self.cross_attn(
                modulate(self.norm2(x), shift_cross, scale_cross),
                cond,
                rope_q=feat_rope,
                rope_k=cond_rope
            )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class MapperBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FrozenDinoV3(nn.Module):
<<<<<<< ours
    def __init__(self, model_name="dinov3_vitl16", repo="facebook/dinov3-vitl16-pretrain-sat493m", pretrained=True):
        super().__init__()
        self.model = torch.hub.load(
            repo,
            model_name,
            source='local',
            path="/data/yjy_data/JiT/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        )

=======
    def __init__(self, model_name="dinov3_vitl14", repo="facebookresearch/dinov3", pretrained=True):
        super().__init__()
        self.model = torch.hub.load(repo, model_name, pretrained=pretrained)
>>>>>>> theirs
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                outputs = self.model.forward_features(x)
            else:
                outputs = self.model(x)
        if isinstance(outputs, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "x_patchtokens", "x_norm_patch_tokens"):
                if key in outputs:
                    tokens = outputs[key]
                    break
            else:
                tokens = outputs.get("last_hidden_state", outputs.get("features", outputs))
        else:
            tokens = outputs
        return tokens


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        mapper_depth=2,
        mapper_mlp_ratio=2.0,
        mapper_attn_drop=0.0,
        mapper_proj_drop=0.0,
        dino_repo="facebookresearch/dinov3",
        dino_model="dinov3_vitl14",
        dino_pretrained=True,
        prototype_path=None
    ):
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )
        self.cond_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # frozen dino encoder + mapper
        self.dino = FrozenDinoV3(model_name=dino_model, repo=dino_repo, pretrained=dino_pretrained)
        self.dino_dim = getattr(self.dino.model, "embed_dim", getattr(self.dino.model, "num_features", hidden_size))
        self.mapper_in = nn.Linear(self.dino_dim, hidden_size, bias=True)
        self.mapper_blocks = nn.ModuleList([
            MapperBlock(hidden_size, num_heads, mlp_ratio=mapper_mlp_ratio,
                        attn_drop=mapper_attn_drop, proj_drop=mapper_proj_drop)
            for _ in range(mapper_depth)
        ])
        self.mapper_out = nn.Linear(hidden_size, self.dino_dim, bias=True)
        self.cond_proj = nn.Linear(self.dino_dim, hidden_size, bias=True)

        self.register_buffer("dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if prototype_path is not None:
            prototypes = torch.load(prototype_path, map_location="cpu")
            if isinstance(prototypes, dict):
                prototypes = prototypes.get("prototypes", prototypes.get("centroids", prototypes))
            self.register_buffer("opt_prototypes", prototypes)
        else:
            self.opt_prototypes = None

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.xavier_uniform_(self.mapper_in.weight)
        nn.init.constant_(self.mapper_in.bias, 0)
        nn.init.xavier_uniform_(self.mapper_out.weight)
        nn.init.constant_(self.mapper_out.bias, 0)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.constant_(self.cond_proj.bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _prepare_dino_inputs(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        elif img.shape[1] != 3:
            raise ValueError(f"DINO expects 3 channels, got {img.shape[1]}")
        img = (img + 1.0) * 0.5
        return (img - self.dino_mean) / self.dino_std

    def _resize_tokens(self, tokens, target_hw):
        bsz, num_tokens, dim = tokens.shape
        src_hw = int(num_tokens ** 0.5)
        if src_hw * src_hw != num_tokens:
            raise ValueError(f"Tokens count {num_tokens} is not a square.")
        if src_hw == target_hw:
            return tokens
        tokens_2d = tokens.view(bsz, src_hw, src_hw, dim).permute(0, 3, 1, 2)
        tokens_2d = F.interpolate(tokens_2d, size=(target_hw, target_hw), mode="bicubic", align_corners=False)
        tokens = tokens_2d.permute(0, 2, 3, 1).reshape(bsz, target_hw * target_hw, dim)
        return tokens

    def _encode_condition(self, sar_img, opt_img=None, return_mapper_loss=False):
        sar_input = self._prepare_dino_inputs(sar_img)
        sar_tokens = self.dino(sar_input)
        if sar_tokens.dim() == 3 and sar_tokens.shape[1] > 1 and int((sar_tokens.shape[1] - 1) ** 0.5) ** 2 == (
            sar_tokens.shape[1] - 1
        ):
            sar_tokens = sar_tokens[:, 1:, :]

        target_hw = self.input_size // self.patch_size
        sar_tokens = self._resize_tokens(sar_tokens, target_hw)

        mapped = self.mapper_in(sar_tokens)
        for block in self.mapper_blocks:
            mapped = block(mapped, feat_rope=self.cond_rope)
        mapped = self.mapper_out(mapped)

        cond_tokens = self.cond_proj(mapped)

        mapper_loss = None
        proto_loss = None
        if return_mapper_loss and opt_img is not None:
            opt_input = self._prepare_dino_inputs(opt_img)
            opt_tokens = self.dino(opt_input)
            if opt_tokens.dim() == 3 and opt_tokens.shape[1] > 1 and int((opt_tokens.shape[1] - 1) ** 0.5) ** 2 == (
                opt_tokens.shape[1] - 1
            ):
                opt_tokens = opt_tokens[:, 1:, :]
            opt_tokens = self._resize_tokens(opt_tokens, target_hw)
            mapper_loss = F.mse_loss(mapped, opt_tokens)

            if self.opt_prototypes is not None:
                mapped_norm = F.normalize(mapped, dim=-1)
                prototypes = F.normalize(self.opt_prototypes, dim=-1)
                sims = torch.einsum("bnd,kd->bnk", mapped_norm, prototypes)
                proto_loss = 1.0 - sims.max(dim=-1).values.mean()
        return cond_tokens, mapper_loss, proto_loss

    def forward(self, x, t, y, sar_img=None, opt_img=None, return_mapper_loss=False):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        cond_tokens = None
        mapper_loss = None
        proto_loss = None
        if sar_img is not None:
            cond_tokens, mapper_loss, proto_loss = self._encode_condition(
                sar_img,
                opt_img=opt_img,
                return_mapper_loss=return_mapper_loss
            )

        # forward JiT
        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(
                x,
                c,
                cond=cond_tokens,
                feat_rope=self.feat_rope if i < self.in_context_start else self.feat_rope_incontext,
                cond_rope=self.cond_rope
            )

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if return_mapper_loss:
            return output, {"mapper_loss": mapper_loss, "proto_loss": proto_loss}
        return output


def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)


def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)


def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)


def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)


def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
