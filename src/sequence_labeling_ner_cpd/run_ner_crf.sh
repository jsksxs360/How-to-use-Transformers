export OUTPUT_DIR=./ner_crf_results/

python3 run_ner_crf.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=bert \
    --model_checkpoint=bert-base-chinese \
    --train_file=../../data/china-people-daily-ner-corpus/example.train \
    --dev_file=../../data/china-people-daily-ner-corpus/example.dev \
    --test_file=../../data/china-people-daily-ner-corpus/example.test \
    --max_seq_length=512 \
    --learning_rate=1e-5 \
    --crf_learning_rate=5e-5 \
    --num_train_epochs=3 \
    --batch_size=4 \
    --do_train \
    --warmup_proportion=0. \
    --seed=42