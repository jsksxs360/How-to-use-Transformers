import os
import logging
import json
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
import sys
sys.path.append('../../')
from src.tools import seed_everything
from src.seq2seq_summarization.arg import parse_args
from src.seq2seq_summarization.data import LCSTS, get_dataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = epoch * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(args.device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, model, tokenizer):
    preds, labels = [], []
    rouge = Rouge()

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(args.device)
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.beam_search_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            label_tokens = batch_data["labels"].cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            preds += [' '.join(pred.strip()) for pred in decoded_preds]
            labels += [' '.join(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    return result

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, model, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, model, tokenizer, shuffle=False)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))
    
    total_loss = 0.
    best_avg_rouge = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + 30 * "-")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        dev_rouges = test_loop(args, dev_dataloader, model, tokenizer)
        logger.info(f"Dev Rouge1: {dev_rouges['rouge-1']:>0.2f} Rouge2: {dev_rouges['rouge-2']:>0.2f} RougeL: {dev_rouges['rouge-l']:>0.2f}")
        rouge_avg = dev_rouges['avg']
        if rouge_avg > best_avg_rouge:
            best_avg_rouge = rouge_avg
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_rouge_avg_{rouge_avg:0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, model, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        test_rouges = test_loop(args, test_dataloader, model, tokenizer)
        logger.info(f"Test Rouge1: {test_rouges['rouge-1']:>0.2f} Rouge2: {test_rouges['rouge-2']:>0.2f} RougeL: {test_rouges['rouge-l']:>0.2f}")

def predict(args, document:str, model, tokenizer):
    inputs = tokenizer(
        document, 
        max_length=args.max_input_length, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = inputs.to(args.device)
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_target_length,
            num_beams=args.beam_search_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        ).cpu().numpy()
    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]
    decoded_preds = tokenizer.decode(
        generated_tokens[0], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return decoded_preds

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    # Training
    if args.do_train:
        # Set seed
        seed_everything(args.seed)
        train_dataset = LCSTS(args.train_file)
        dev_dataset = LCSTS(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = LCSTS(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = LCSTS(args.test_file)
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            logger.info(f'predicting labels of {save_weight}...')

            results = []
            model.eval()
            for s_idx in tqdm(range(len(test_dataset))):
                sample = test_dataset[s_idx]
                pred_summ = predict(args, sample['content'], model, tokenizer)
                results.append({
                    "sentence": sample['content'], 
                    "prediction": pred_summ, 
                    "summarization": sample['title']
                })
            with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
                for exapmle_result in results:
                    f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
