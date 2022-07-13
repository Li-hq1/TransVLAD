OUTPUT_DIR='output/pretrain/mini'
DATASET_NAME='mini' #'mini','CIFAR_FS','FC100','tiered','CUB'
GPUS='1,3'
DIST_URL='tcp://127.0.0.1:6666'
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 6666 run_pretrain.py \
        --dist_url ${DIST_URL} \
        --gpus ${GPUS} \
        --dataset_name ${DATASET_NAME} \
        --mask_ratio 0.75 \
        --model pretrain_mae_base_patch16_224 \
        --batch_size 128 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --output_dir ${OUTPUT_DIR}
