# meta test
MODEL_PATH='output/finetune/MINI_1600/checkpoint-best_meta_val.pth'
DATASET_NAME='mini'  #'mini', 'FC100', 'tiered', 'CUB', 'CIFAR_FS'
OUTPUT_DIR='output/finetune_output/test'
GPUS='0'
python run_finetune.py \
    --seed 2 \
    --no_auto_resume \
    --gpus ${GPUS} \
    --model vit_base_patch16_224 \
    --dataset_name ${DATASET_NAME} \
    --finetune ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --pooling nextvlad \
    --meta_distance cos \
    --mask_ratio 0 \
    --meta_test \
    --meta_val \
    --nextvlad_lamb 4 \
    --nextvlad_cluster 64 \
    --nextvlad_groups 8 \
