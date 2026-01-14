import sys
sys.path.append("/data/yjy_data/FSPCG")
import argparse
import copy
import math
import os
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model_jit import JiT_models
from util.datasets import ImageDirDataset
from util.lora import LoRALinear


class RandomRotate90:
    def __call__(self, img):
        k = torch.randint(0, 4, (1,)).item()
        if k == 0:
            return img
        return transforms.functional.rotate(img, angle=90 * k)


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        tensor = transforms.functional.to_tensor(img)
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return transforms.functional.to_pil_image(tensor)


class MultiCropTransform:
    def __init__(
        self,
        global_crop_size,
        local_crop_size,
        global_crops_number=2,
        local_crops_number=6,
    ):
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([blur], p=0.3),
            transforms.RandomApply([AddGaussianNoise(std=0.02)], p=0.3),
            transforms.ToTensor(),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=(0.2, 0.6), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([blur], p=0.3),
            transforms.RandomApply([AddGaussianNoise(std=0.02)], p=0.3),
            transforms.ToTensor(),
        ])

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

    def __call__(self, img):
        crops = [self.global_transform(img) for _ in range(self.global_crops_number)]
        crops.extend(self.local_transform(img) for _ in range(self.local_crops_number))
        return crops


def collate_multi_crop(batch):
    views = list(zip(*batch))
    return [torch.stack(view, dim=0) for view in views]


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


@dataclass
class TeacherTempSchedule:
    base_temp: float = 0.04
    final_temp: float = 0.07
    warmup_epochs: int = 30
    total_epochs: int = 300

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.base_temp + (self.final_temp - self.base_temp) * epoch / max(1, self.warmup_epochs)
        return self.final_temp


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_out = student_out / self.student_temp
        student_logprob = F.log_softmax(student_out, dim=-1)

        teacher_out = teacher_out.detach()
        teacher_out = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1)
        loss = torch.sum(-teacher_out * student_logprob, dim=-1).mean()

        new_center = torch.mean(teacher_out, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + new_center * (1 - self.center_momentum)
        return loss

    def set_teacher_temp(self, temp):
        self.teacher_temp = temp


def freeze_non_lora(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.lora_parameters():
                param.requires_grad = True


def build_args():
    parser = argparse.ArgumentParser("JiT LoRA SSL", add_help=False)
    parser.add_argument("--data_path", default="/NAS_data/yjy/S2O_data_all_dataset/three_datasets", type=str, help="Path to optical image dataset")
    parser.add_argument("--output_dir", default="/data/yjy_data/FSPCG/JiT_opt_pretrained", type=str)
    parser.add_argument("--jit_ckpt_path", default="/data/yjy_data/JiT/checkpoint-last.pth", type=str, help="Path to pretrained JiT checkpoint (.pth)")
    parser.add_argument("--model", default="JiT-B/16", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--global_crop_size", default=256, type=int, choices=[224, 256])
    parser.add_argument("--local_crop_size", default=96, type=int)
    parser.add_argument("--global_crops_number", default=2, type=int)
    parser.add_argument("--local_crops_number", default=6, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--out_dim", default=65536, type=int)
    parser.add_argument("--ibot", action="store_true", help="Enable patch-level self-distillation")
    parser.add_argument("--ibot_weight", default=0.5, type=float)
    parser.add_argument("--device", default="cuda:7", type=str)
    return parser


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    cudnn.benchmark = True

    transform = MultiCropTransform(
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
        global_crops_number=args.global_crops_number,
        local_crops_number=args.local_crops_number,
    )
    dataset = ImageDirDataset(args.data_path, transform=transform, mode="RGB")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_multi_crop,
    )

    student = JiT_models[args.model](
        input_size=args.img_size,
        in_channels=3,
        out_channels=3,
        num_classes=2,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_dino=False,
    )
    if args.jit_ckpt_path:
        checkpoint = torch.load(args.jit_ckpt_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("student", checkpoint))
        missing, unexpected = student.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[JiT SSL] Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    teacher = copy.deepcopy(student)
    student_head = DINOHead(student.hidden_size, out_dim=args.out_dim)
    teacher_head = copy.deepcopy(student_head)
    patch_head = DINOHead(student.hidden_size, out_dim=args.out_dim) if args.ibot else None
    patch_head_teacher = copy.deepcopy(patch_head) if args.ibot else None

    student.to(device)
    teacher.to(device)
    student_head.to(device)
    teacher_head.to(device)
    if args.ibot:
        patch_head.to(device)
        patch_head_teacher.to(device)

    teacher.eval()
    teacher_head.eval()
    if args.ibot:
        patch_head_teacher.eval()

    freeze_non_lora(student)

    trainable_params = [
        param for param in student.parameters() if param.requires_grad
    ] + list(student_head.parameters())
    if args.ibot:
        trainable_params += list(patch_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    temp_schedule = TeacherTempSchedule(total_epochs=args.epochs)
    dino_loss = DINOLoss(args.out_dim).to(device)

    momentum_base = 0.996
    for epoch in range(args.epochs):
        student.train()
        student_head.train()
        if args.ibot:
            patch_head.train()
        dino_loss.set_teacher_temp(temp_schedule(epoch))

        for crops in loader:
            crops = [crop.to(device, non_blocking=True) for crop in crops]
            global_views = crops[: args.global_crops_number]
            all_views = crops

            student_outputs = []
            student_tokens = []
            for view in all_views:
                feats, tokens = student.encode_features(view, return_tokens=True)
                student_outputs.append(student_head(feats))
                student_tokens.append(tokens)

            with torch.no_grad():
                teacher_outputs = []
                teacher_tokens = []
                for view in global_views:
                    feats, tokens = teacher.encode_features(view, return_tokens=True)
                    teacher_outputs.append(teacher_head(feats))
                    teacher_tokens.append(tokens)

            total_loss = 0.0
            n_loss_terms = 0
            for t_idx, t_out in enumerate(teacher_outputs):
                for s_idx, s_out in enumerate(student_outputs):
                    if s_idx == t_idx:
                        continue
                    total_loss = total_loss + dino_loss(s_out, t_out)
                    n_loss_terms += 1
            total_loss = total_loss / max(1, n_loss_terms)

            if args.ibot:
                ibot_loss = 0.0
                ibot_terms = 0
                for t_idx, t_tokens in enumerate(teacher_tokens):
                    t_patch = patch_head_teacher(t_tokens.reshape(-1, t_tokens.shape[-1]))
                    for s_idx, s_tokens in enumerate(student_tokens):
                        if s_idx == t_idx:
                            continue
                        s_patch = patch_head(s_tokens.reshape(-1, s_tokens.shape[-1]))
                        ibot_loss = ibot_loss + dino_loss(s_patch, t_patch)
                        ibot_terms += 1
                ibot_loss = ibot_loss / max(1, ibot_terms)
                total_loss = total_loss + args.ibot_weight * ibot_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                momentum = 1 - (1 - momentum_base) * (math.cos(math.pi * epoch / args.epochs) + 1) / 2
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)
                for s_param, t_param in zip(student_head.parameters(), teacher_head.parameters()):
                    t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)
                if args.ibot:
                    for s_param, t_param in zip(patch_head.parameters(), patch_head_teacher.parameters()):
                        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)

        ckpt = {
            "student": student.state_dict(),
            "student_head": student_head.state_dict(),
            "epoch": epoch,
        }
        if args.ibot:
            ckpt["patch_head"] = patch_head.state_dict()
        torch.save(ckpt, os.path.join(args.output_dir, f"jit_ssl_epoch_{epoch:04d}.pth"))


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
