## Just image Transformer (JiT) for SAR-to-Optical Image Translation

Train：
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 main_jit.py

Inference：
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py \
  --evaluate_gen \
  --resume /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt \
  --sar_test_path /data/yjy_data/dataset/SAR2Opt/test/A \
  --output_dir /NAS_data/hjf/JiTcolor/outputs/SAR2Opt \
  --img_size 512 \
  --gen_bsz 8
