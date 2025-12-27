## Just image Transformer (JiT) for SAR-to-Optical Image Translation

Train：
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 main_jit.py

Train with different lr:
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=29505 main_jit.py \
  --blr 1.6e-3 \
  --output_dir /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt/lr5em5

Train on GF3:
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master-port=29506 main_jit.py --output_dir "/NAS_data/hjf/JiTcolor/checkpoints/GF3" --sar_train_path="/NAS_data/yjy/GF3_High_Res/trainA" --opt_train_path="/NAS_data/yjy/GF3_High_Res/trainB" --sar_test_path="/NAS_data/yjy/GF3_High_Res/testA" --opt_test_path="/NAS_data/yjy/GF3_High_Res/testB" --img_size=256

Inference：

SAR2Opt：
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29503 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/SAR2Opt --sar_test_path /NAS_data/yjy/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/SAR2Opt/round1 --img_size 512 --gen_bsz 8

GF3：
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=29504 main_jit.py --evaluate_gen --resume /NAS_data/hjf/JiTcolor/checkpoints/GF3 --sar_test_path /NAS_data/yjy/GF3_High_Res/testA --output_dir /NAS_data/hjf/JiTcolor/outputs/GF3/round1 --img_size 256 --gen_bsz 8
