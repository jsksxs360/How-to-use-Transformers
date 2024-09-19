export OUTPUT_DIR=./trans_marian_results/

python3 run_translation_marian.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=marian \
    --model_checkpoint=Helsinki-NLP/opus-mt-zh-en \
    --train_file=../../data/translation2019zh/translation2019zh_train.json \
    --dev_file=../../data/translation2019zh/translation2019zh_train.json \
    --test_file=../../data/translation2019zh/translation2019zh_valid.json \
    --max_length=128 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=32 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42