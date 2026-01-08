import math
import sys
import os
import shutil

import torchvision.transforms as transforms

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy

from util.datasets import ImageDirDataset
import numpy as _np
import torch
from PIL import Image


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (sar_img, opt_img) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        sar_img = sar_img.to(device, non_blocking=True).to(torch.float32).div_(255)
        sar_img = sar_img * 2.0 - 1.0
        opt_img = opt_img.to(device, non_blocking=True).to(torch.float32).div_(255)
        opt_img = opt_img * 2.0 - 1.0
        labels = torch.zeros(opt_img.size(0), device=device, dtype=torch.long)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(opt_img, sar_img, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    distributed = misc.is_dist_avail_and_initialized()

    transform_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.PILToTensor()
    ])
    sar_dataset = ImageDirDataset(args.sar_test_path, transform=transform_eval, mode="L")
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            sar_dataset, num_replicas=world_size, rank=local_rank, shuffle=False
        )
    else:
        sampler = torch.utils.data.SequentialSampler(sar_dataset)
    data_loader = torch.utils.data.DataLoader(
        sar_dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    num_images = min(args.num_images, len(sar_dataset))
    num_steps = (num_images // (batch_size * world_size)) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    img_count = 0
    def _to_uint8_image(img):
        if torch.is_tensor(img):
            img = img.detach().cpu().float().numpy()
        elif not isinstance(img, _np.ndarray):
            img = _np.asarray(img)

        if img.dtype == _np.object_:
            img = img.astype(_np.float32)

        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
            img = img.transpose(1, 2, 0)

        img = _np.clip(img, 0, 1)
        img = (img * 255).round().astype(_np.uint8)

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

    for i, (sar_img, sar_names) in enumerate(data_loader):
        if img_count >= num_images:
            break
        print("Generation step {}/{}".format(i, num_steps))

        sar_img = sar_img.to(torch.device(args.device))
        sar_img = sar_img.to(torch.float32).div_(255)
        sar_img = sar_img * 2.0 - 1.0
        labels_gen = torch.zeros(sar_img.size(0), device=sar_img.device, dtype=torch.long)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(sar_img, labels_gen)

        if distributed:
            torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        # sampled_images = sampled_images.detach().cpu()
        sampled_images = sampled_images.detach().cpu().float()
        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = img_count + b_id
            if img_id >= num_images:
                break
            gen_img = _to_uint8_image(sampled_images[b_id])

            # 路径 & 后缀
            name = str(sar_names[b_id])
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                name += ".png"

            out_path = os.path.join(save_folder, name)

            _save_image(gen_img, out_path)

    if distributed:
        torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not args.keep_outputs:
            shutil.rmtree(save_folder)

    if distributed:
        torch.distributed.barrier()
