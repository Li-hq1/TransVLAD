PRETRIAN_MODEL_CHECKPOINT='output/pretrain/MINI_1600/checkpoint-1599.pth'
DATASET_NAME='mini' #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/finetune/MINI_1600'
GPUS='0' # we use sinle GPU
python run_finetune.py \
    --lr 7e-4 \
    --seed 0 \
    --gpus ${GPUS} \
    --model vit_base_patch16_224 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${PRETRIAN_MODEL_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --pooling nextvlad \
    --mask_ratio 0.7 \
    --focal_gamma 2 \
    --meta_distance cos \
    --meta_val \
    --nextvlad_lamb 4 \
    --nextvlad_cluster 64 \
    --nextvlad_groups 8 \