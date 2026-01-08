# 无条件采样
# python unconditional_generate.py \
#   --mode sample \
#   --resume /path/to/checkpoint-dir \
#   --output_dir /path/to/output \
#   --num_images 100 \
#   --gen_bsz 8 \
#   --img_size 512 \
#   --sar_concat_mode none

# 如需使用 EMA 权重生成
# python unconditional_generate.py \
#   --mode sample \
#   --resume /path/to/checkpoint-dir \
#   --use_ema \
#   --sar_concat_mode none

#  基于 checkpoint 再训练（finetune）
# python unconditional_generate.py \
#   --mode train \
#   --resume /path/to/checkpoint-dir \
#   --opt_train_path /path/to/opt_train_images \
#   --output_dir /path/to/output \
#   --epochs 10 \
#   --batch_size 8 \
#   --img_size 512 \
#   --sar_concat_mode none

# 如果希望从 checkpoint 继续优化器状态（包括学习率、动量等）
# python unconditional_generate.py \
#   --mode train \
#   --resume /path/to/checkpoint-dir \
#   --resume_optimizer \
#   --opt_train_path /path/to/opt_train_images \
#   --sar_concat_mode none
import argparse
import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from denoiser import Denoiser
from main_jit import get_args_parser
import util.misc as misc
from util.datasets import ImageDirDataset


def _to_uint8_image(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().float().numpy()
    elif not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.dtype == np.object_:
        img = img.astype(np.float32)

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
        img = img.transpose(1, 2, 0)

    img = np.clip(img, 0, 1)
    img = (img * 255).round().astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    if img.ndim not in (2, 3):
        raise ValueError(f"Unexpected image shape: {img.shape}")

    return img


def _save_image(img, path):
    if img.ndim == 2:
        Image.fromarray(img, mode="L").save(path)
    elif img.shape[2] == 3:
        Image.fromarray(img, mode="RGB").save(path)
    else:
        raise ValueError(f"Unsupported image shape for saving: {img.shape}")


@torch.no_grad()
def _unconditional_step(model, z, t):
    labels = torch.full((z.size(0),), model.num_classes, device=z.device, dtype=torch.long)
    x_pred = model.net(z, t.flatten(), labels, sar_img=None)
    v_pred = (x_pred - z) / (1.0 - t).clamp_min(model.t_eps)
    return v_pred


@torch.no_grad()
def sample_unconditional(model, batch_size, device):
    z = model.noise_scale * torch.randn(batch_size, 3, model.img_size, model.img_size, device=device)
    timesteps = torch.linspace(0.0, 1.0, model.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(
        -1, batch_size, -1, -1, -1
    )

    if model.method == "euler":
        def stepper(cur_z, t, t_next):
            v_pred = _unconditional_step(model, cur_z, t)
            return cur_z + (t_next - t) * v_pred
    elif model.method == "heun":
        def stepper(cur_z, t, t_next):
            v_pred_t = _unconditional_step(model, cur_z, t)
            z_next_euler = cur_z + (t_next - t) * v_pred_t
            v_pred_t_next = _unconditional_step(model, z_next_euler, t_next)
            v_pred = 0.5 * (v_pred_t + v_pred_t_next)
            return cur_z + (t_next - t) * v_pred
    else:
        raise NotImplementedError(f"Unknown sampling method: {model.method}")

    for i in range(model.steps - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        z = stepper(z, t, t_next)
    v_pred = _unconditional_step(model, z, timesteps[-2])
    z = z + (timesteps[-1] - timesteps[-2]) * v_pred
    return z


def load_checkpoint(model, resume_path, use_ema=True):
    if resume_path is None:
        return None
    if os.path.isdir(resume_path):
        checkpoint_path = os.path.join(resume_path, "checkpoint-last.pth")
    else:
        checkpoint_path = resume_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if use_ema and "model_ema1" in checkpoint:
        model.load_state_dict(checkpoint["model_ema1"])
    else:
        model.load_state_dict(checkpoint["model"])
    return checkpoint


def build_output_dir(args, model):
    suffix = "unconditional-steps{}-image{}-res{}".format(model.steps, args.num_images, args.img_size)
    return os.path.join(args.output_dir, suffix)


def get_parser():
    parser = argparse.ArgumentParser("JiT unconditional generation", parents=[get_args_parser()])
    parser.add_argument("--use_ema", action="store_true", help="Load EMA weights if available")
    parser.add_argument("--mode", default="sample", choices=["sample", "train"], help="Run sampling or finetuning")
    parser.add_argument("--resume_optimizer", action="store_true", help="Load optimizer state when resuming training")
    return parser


def run_sampling(args, model, device):
    model.eval()
    save_folder = build_output_dir(args, model)
    os.makedirs(save_folder, exist_ok=True)

    num_images = args.num_images
    batch_size = args.gen_bsz
    img_count = 0

    while img_count < num_images:
        cur_batch = min(batch_size, num_images - img_count)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            sampled_images = sample_unconditional(model, cur_batch, device)

        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu().float()
        for b_id in range(sampled_images.size(0)):
            img_id = img_count + b_id
            gen_img = _to_uint8_image(sampled_images[b_id])
            out_path = os.path.join(save_folder, f"uncond_{img_id:05d}.png")
            _save_image(gen_img, out_path)
        img_count += sampled_images.size(0)


def run_training(args, model, device, checkpoint):
    model.train(True)

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.PILToTensor(),
    ])
    dataset_train = ImageDirDataset(args.opt_train_path, transform=transform_train, mode="RGB")
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=True,
        drop_last=True,
    )

    eff_batch_size = args.batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if checkpoint is not None and args.resume_optimizer:
        if "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    if checkpoint is not None and "model_ema1" in checkpoint and "model_ema2" in checkpoint:
        ema_state_dict1 = checkpoint["model_ema1"]
        ema_state_dict2 = checkpoint["model_ema2"]
        model.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model.named_parameters()]
        model.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model.named_parameters()]
    else:
        model.ema_params1 = [p.detach().clone() for p in model.parameters()]
        model.ema_params2 = [p.detach().clone() for p in model.parameters()]

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        for data_iter_step, (opt_img, _names) in enumerate(data_loader):
            opt_img = opt_img.to(device, non_blocking=True).to(torch.float32).div_(255)
            opt_img = opt_img * 2.0 - 1.0
            labels = torch.zeros(opt_img.size(0), device=device, dtype=torch.long)

            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                loss = model(opt_img, sar_img=None, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            if data_iter_step % args.log_freq == 0:
                print(f"Epoch {epoch} iter {data_iter_step}: loss {loss.item():.6f}")

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.sar_concat_mode != "none":
        raise ValueError("Unconditional generation requires --sar_concat_mode none.")

    device = torch.device(args.device)
    model = Denoiser(args).to(device)

    checkpoint = load_checkpoint(model, args.resume, use_ema=args.use_ema)

    if args.mode == "train":
        run_training(args, model, device, checkpoint)
    else:
        run_sampling(args, model, device)


if __name__ == "__main__":
    main()
