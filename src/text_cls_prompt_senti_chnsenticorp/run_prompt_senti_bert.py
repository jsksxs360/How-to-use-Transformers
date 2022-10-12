import os
import json
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.text_cls_prompt_senti_chnsenticorp.data import ChnSentiCorp, get_dataLoader, POS_TOKEN, NEG_TOKEN, PROMPT
from src.text_cls_prompt_senti_chnsenticorp.modeling import BertForPrompt
from src.text_cls_prompt_senti_chnsenticorp.arg import parse_args
from src.tools import seed_everything

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, dataset, model, tokenizer):
    results = []
    pos_id = tokenizer.convert_tokens_to_ids(POS_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NEG_TOKEN)
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            token_logits = outputs[1]
            mask_token_indexs = torch.where(batch_data["batch_inputs"]["input_ids"] == tokenizer.mask_token_id)[1]
            for s_idx, mask_idx in enumerate(mask_token_indexs):
                results.append(token_logits[s_idx, mask_idx, [neg_id, pos_id]].cpu().numpy())
        true_labels = [
            int(dataset[s_idx]['label']) for s_idx in range(len(dataset))
        ]
        predictions = np.asarray(results).argmax(axis=-1).tolist()
    return classification_report(true_labels, predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)
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
    best_f1_score = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, dev_dataset, model, tokenizer)
        macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
        dev_f1_score = (macro_f1 + micro_f1) / 2
        logger.info(f'Dev: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f}')
        if dev_f1_score > best_f1_score:
            best_f1_score = dev_f1_score
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_macrof1_{(100*macro_f1):0.4f}_microf1_{(100*micro_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, test_dataset, model, tokenizer)
        pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
        neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
        macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
        logger.info(f'POS: {100*pos_p:>0.2f} / {100*pos_r:>0.2f} / {100*pos_f1:>0.2f}, NEG: {100*neg_p:>0.2f} / {100*neg_r:>0.2f} / {100*neg_f1:>0.2f}')
        logger.info(f'Test: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f}')

def predict(args, comment:str, model, tokenizer):
    inputs = tokenizer(
        PROMPT(comment), 
        max_length=args.max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(args, inputs)
    pos_id = tokenizer.convert_tokens_to_ids(POS_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NEG_TOKEN)
    
    with torch.no_grad():
        outputs = model(**inputs)
        token_logits = outputs[1]
        pred = token_logits[0, mask_token_index, [neg_id, pos_id]].cpu().numpy()
        probs = torch.nn.functional.softmax(torch.tensor(pred), dim=-1)
    return {
        "pred": '1' if probs[1] > probs[0] else '0', 
        "prediction": {'0': probs[0].item(), '1': probs[1].item()}
    }

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
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertForPrompt.from_pretrained(
        args.model_checkpoint,
        config=config
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = ChnSentiCorp(args.train_file)
        dev_dataset = ChnSentiCorp(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = ChnSentiCorp(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = ChnSentiCorp(args.test_file)
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            logger.info(f'predicting labels of {save_weight}...')
            
            results = []
            model.eval()
            for s_idx in tqdm(range(len(test_dataset))):
                sample = test_dataset[s_idx]
                pred_res = predict(args, sample['comment'], model, tokenizer)
                results.append({
                    "comment": sample['comment'], 
                    "true_label": sample['label'], 
                    "pred_label": pred_res['pred'], 
                    "prediction": pred_res['prediction']
                })
            with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
                    for exapmle_result in results:
                        f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')