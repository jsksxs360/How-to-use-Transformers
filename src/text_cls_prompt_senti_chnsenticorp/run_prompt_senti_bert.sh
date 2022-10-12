export OUTPUT_DIR=./prompt_senti_bert_results/

python3 run_prompt_senti_bert.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/ChnSentiCorp/train.txt \
    --dev_file=../../data/ChnSentiCorp/dev.txt \
    --test_file=../../data/ChnSentiCorp/test.txt \
    --max_length=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=12