import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model_jit import JiT_models
from util.datasets import ImageDirDataset


def build_args():
    parser = argparse.ArgumentParser("JiT SSL Inference", add_help=False)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--model", default="JiT-B/16", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser


def main(args):
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    dataset = ImageDirDataset(args.data_path, transform=transform, mode="RGB")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = JiT_models[args.model](
        input_size=args.img_size,
        in_channels=3,
        out_channels=3,
        num_classes=2,
        use_dino=False,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("student", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    features = []
    names = []
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device, non_blocking=True)
            feats = model.encode_features(images)
            features.append(feats.cpu().numpy())
            names.extend(filenames)

    features = np.concatenate(features, axis=0)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, features=features, names=np.array(names))


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
