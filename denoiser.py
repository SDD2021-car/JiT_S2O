import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models
from sar_encoder import SAREncoder
from subspace_head import SubspaceHead


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        sar_concat_mode = getattr(args, "sar_concat_mode", "none")
        sar_concat_channels = getattr(args, "sar_concat_channels", 1)
        use_dino = not getattr(args, "disable_dino", False)
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            mapper_depth=args.mapper_depth,
            mapper_mlp_ratio=args.mapper_mlp_ratio,
            mapper_attn_drop=args.mapper_attn_drop,
            mapper_proj_drop=args.mapper_proj_drop,
            dino_repo=args.dino_repo,
            dino_model=args.dino_model,
            dino_pretrained=args.dino_pretrained,
            dino_ckpt_path=args.dino_ckpt_path,
            prototype_path=args.prototype_path,
            sar_concat_mode=sar_concat_mode,
            sar_concat_channels=sar_concat_channels,
            use_dino=use_dino,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.subspace_enabled = getattr(args, "enable_subspace", False)
        self.subspace_reg_lambda = getattr(args, "subspace_reg_lambda", 1e-4)

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.k = getattr(args, "k", self.cfg_scale)
        gamma_min = getattr(args, "gamma_min", self.cfg_interval[0])
        gamma_max = getattr(args, "gamma_max", self.cfg_interval[1])
        self.gamma_schedule = (gamma_min, gamma_max)
        self.energy_tau = getattr(args, "energy_tau", 0.0)

        self.mapper_loss_weight = args.mapper_loss_weight
        self.proto_loss_weight = args.prototype_loss_weight

        if self.subspace_enabled:
            self.sar_encoder = SAREncoder(
                input_size=args.img_size,
                patch_size=self.net.patch_size,
                in_channels=getattr(args, "subspace_in_channels", 1),
                embed_dim=self.net.hidden_size,
                token_count=self.net.x_embedder.num_patches,
                scheme=getattr(args, "subspace_scheme", "B"),
            )
            self.subspace_head = SubspaceHead(
                embed_dim=self.net.hidden_size,
                rank_k=getattr(args, "subspace_rank", 16),
            )
            self.subspace_match_weight = nn.Parameter(
                torch.tensor(getattr(args, "subspace_match_weight_init", 1.0))
            )
            self.subspace_ortho_weight = nn.Parameter(
                torch.tensor(getattr(args, "subspace_ortho_weight_init", 0.1))
            )
            self.prior_net = copy.deepcopy(self.net)
            prior_ckpt_path = getattr(args, "prior_ckpt_path", None)
            if prior_ckpt_path:
                prior_checkpoint = torch.load(prior_ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(prior_checkpoint, dict) and "model" in prior_checkpoint:
                    prior_state = prior_checkpoint["model"]
                else:
                    prior_state = prior_checkpoint
                if isinstance(prior_state, dict):
                    has_net_prefix = any(key.startswith("net.") for key in prior_state.keys())
                    if has_net_prefix:
                        prior_state = {key.replace("net.", "", 1): value for key, value in prior_state.items()}
                self.prior_net.load_state_dict(prior_state, strict=True)
            self.prior_net.eval()
            for param in self.prior_net.parameters():
                param.requires_grad = False

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.subspace_enabled:
            self.prior_net.eval()
        return self

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, opt_img, sar_img, labels=None):
        if labels is None:
            labels = torch.zeros(opt_img.size(0), device=opt_img.device, dtype=torch.long)
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(opt_img.size(0), device=opt_img.device).view(-1, *([1] * (opt_img.ndim - 1)))
        e = torch.randn_like(opt_img) * self.noise_scale

        z = t * opt_img + (1 - t) * e
        v = (opt_img - z) / (1 - t).clamp_min(self.t_eps)

        net_out = self.net(
            z,
            t.flatten(),
            labels_dropped,
            sar_img=sar_img,
            opt_img=opt_img,
            return_mapper_loss=self.training
        )
        if self.training:
            x_pred, mapper_losses = net_out
        else:
            x_pred = net_out
            mapper_losses = None
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        if self.training and mapper_losses is not None:
            mapper_loss = mapper_losses.get("mapper_loss")
            proto_loss = mapper_losses.get("proto_loss")
            if mapper_loss is not None:
                loss = loss + self.mapper_loss_weight * mapper_loss
            if proto_loss is not None:
                loss = loss + self.proto_loss_weight * proto_loss

        if self.training and self.subspace_enabled:
            if sar_img is None:
                raise ValueError("Subspace training requires sar_img inputs.")
            with torch.no_grad():
                prior_labels = torch.full_like(labels, self.num_classes)
                prior_x = self.prior_net(z, t.flatten(), prior_labels, sar_img=None)
                v_prior = (prior_x - z) / (1 - t).clamp_min(self.t_eps)
                v_tokens = self.net.x_embedder(v)
                v_prior_tokens = self.net.x_embedder(v_prior)
            _, pyramid = self.sar_encoder(sar_img, return_pyramid=True)
            bmat = self.subspace_head(pyramid["proj"])
            v_proj = self.subspace_head.project_tokens(
                v_prior_tokens,
                bmat,
                reg_lambda=self.subspace_reg_lambda,
            )
            token_loss = F.mse_loss(v_proj, v_tokens)
            ortho_loss = self.subspace_head.orthogonality_loss(bmat)
            match_weight = F.softplus(self.subspace_match_weight)
            ortho_weight = F.softplus(self.subspace_ortho_weight)
            loss = loss + match_weight * token_loss + ortho_weight * ortho_loss

        return loss

    @torch.no_grad()
    def generate(self, sar_img, labels=None):
        if labels is None:
            labels = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)
        device = sar_img.device
        bsz = sar_img.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, sar_img)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, sar_img)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, sar_img):
        if self.subspace_enabled:
            return self._forward_subspace(z, t, labels, sar_img)
        # conditional
        x_cond = self.net(z, t.flatten(), labels, sar_img=sar_img)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), sar_img=sar_img)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _forward_subspace(self, z, t, labels, sar_img):
        prior_labels = torch.full_like(labels, self.num_classes)
        prior_x = self.prior_net(z, t.flatten(), prior_labels, sar_img=None)
        v_prior = (prior_x - z) / (1.0 - t).clamp_min(self.t_eps)
        if sar_img is None:
            return v_prior

        v_proj = self._project_prior_velocity(v_prior, t, prior_labels, sar_img)
        v_new = v_proj + self.energy_tau * (v_prior - v_proj)

        low, high = self.gamma_schedule
        interval_mask = (t < high) & ((low == 0) | (t > low))
        k_value = torch.full_like(t, float(self.k))
        k_value = torch.where(interval_mask, k_value, torch.zeros_like(t))
        return v_prior + k_value * (v_new - v_prior)

    @torch.no_grad()
    def _project_prior_velocity(self, v_prior, t, prior_labels, sar_img):
        v_tokens = self.prior_net.x_embedder(v_prior)
        _, pyramid = self.sar_encoder(sar_img, return_pyramid=True)
        bmat = self.subspace_head(pyramid["proj"])
        v_proj_tokens = self.subspace_head.project_tokens(
            v_tokens,
            bmat,
            reg_lambda=self.subspace_reg_lambda,
        )
        c = self.prior_net.t_embedder(t.flatten()) + self.prior_net.y_embedder(prior_labels)
        v_proj_patches = self.prior_net.final_layer(v_proj_tokens, c)
        return self.prior_net.unpatchify(v_proj_patches, self.prior_net.patch_size)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, sar_img):
        v_pred = self._forward_sample(z, t, labels, sar_img)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, sar_img):
        v_pred_t = self._forward_sample(z, t, labels, sar_img)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, sar_img)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
