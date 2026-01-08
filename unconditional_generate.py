import argparse
import os

import numpy as np
from PIL import Image
import torch

from denoiser import Denoiser
from main_jit import get_args_parser


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
        return
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


def build_output_dir(args, model):
    suffix = "unconditional-steps{}-image{}-res{}".format(model.steps, args.num_images, args.img_size)
    return os.path.join(args.output_dir, suffix)


def get_parser():
    parser = argparse.ArgumentParser("JiT unconditional generation", parents=[get_args_parser()])
    parser.add_argument("--use_ema", action="store_true", help="Load EMA weights if available")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.sar_concat_mode != "none":
        raise ValueError("Unconditional generation requires --sar_concat_mode none.")

    device = torch.device(args.device)
    model = Denoiser(args).to(device)
    model.eval()

    load_checkpoint(model, args.resume, use_ema=args.use_ema)

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


if __name__ == "__main__":
    main()
