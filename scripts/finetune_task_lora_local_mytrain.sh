#!/bin/bash
set -x

#!/bin/bash
set -e -x

######### Edit this part #########
export CUDA_VISIBLE_DEVICES=5,6,7
####################################
FINETUNE_DATASET=$1
FINETUNE_MODEL_NAME=$2
BASE_MODEL_PATH=../models/llama3-llava-next-8b


# preprocessing dataset
if [ ! -d "$FINETUNE_DATASET" ]; then
	echo "require a valid directory as the first argument"
	exit 1
fi

[ -L "augmented_rlbench" ] && rm augmented_rlbench
ln -s $FINETUNE_DATASET augmented_rlbench
if [ ! -d "playground" ]; then
	rm playground
fi
python gather_racer_llava_data.py


# train
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print NF}')
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
EPOCH=2

root_path=$(pwd)

torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port 29503 \
    llava/train/my_train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version llava_llama_3_racer \
    --data_path $root_path/playground/racer_llava_data/all_tasks.json \
    --image_folder $root_path \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/${FINETUNE_MODEL_NAME} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1e5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.08 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 3 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name ${FINETUNE_MODEL_NAME} \
    --lang_level simple
