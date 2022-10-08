export OUTPUT_DIR=./extractiveQA_bert_results/

python3 run_extractiveQA.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/cmrc2018/cmrc2018_train.json \
    --dev_file=../../data/cmrc2018/cmrc2018_dev.json \
    --test_file=../../data/cmrc2018/cmrc2018_trial.json \
    --max_length=384 \
    --max_answer_length=30 \
    --stride=128 \
    --n_best=20 \
    --learning_rate=1e-5 \
    --num_train_epochs=3 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42