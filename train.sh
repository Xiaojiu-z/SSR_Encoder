export MODEL_DIR="" # your SD path
export OUTPUT_DIR="SSR_model"  # your save path
cd  SSR  # cd to your path

accelerate launch train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --reg=0.001 \  # 0.001 is good for fine-tuning your model
    --image_column="file_name" \  # image path
    --caption_column="text" \  # caption
    --tag_column="text" \  # caption
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --train_data_dir "./example_data.jsonl" \  # you can use josnl file to save your caption and image paths or make dataset/dataloader by yourself
    --resolution=512 \
    --learning_rate=5e-5 \
    --train_batch_size=24 \
    --num_validation=2 \
    --validation_image './test_img/t2i/1.jpg' \
    --validation_prompts  "a girl" "a cat" \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10000 \
    --validation_steps=10000 \
    --checkpointing_steps=10000