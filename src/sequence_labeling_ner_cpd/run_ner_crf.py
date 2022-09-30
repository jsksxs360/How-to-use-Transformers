import os
import json
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import sys
sys.path.append('../../')
from src.sequence_labeling_ner_cpd.data import PeopleDaily, get_dataLoader, CATEGORIES
from src.sequence_labeling_ner_cpd.modeling import BertCrfForNER
from src.sequence_labeling_ner_cpd.arg import parse_args
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

def test_loop(args, dataloader, model):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]
            tags = model.crf.decode(logits, batch_data['batch_inputs']['attention_mask'])
            predictions = tags.squeeze(0).cpu().numpy()
            labels = batch_data['labels'].cpu().numpy()
            lens = np.sum(batch_data['batch_inputs']['attention_mask'].cpu().numpy(), axis=-1)
            true_labels += [
                [args.id2label[int(l)] for idx, l in enumerate(label) if idx > 0 and idx < seq_len - 1] 
                for label, seq_len in zip(labels, lens)
            ]
            true_predictions += [
                [args.id2label[int(p)] for idx, p in enumerate(prediction) if idx > 0 and idx < seq_len - 1]
                for prediction, seq_len in zip(predictions, lens)
            ]
    return classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    cls_param_optimizer = list(model.classifier.named_parameters())
    if args.use_ffnn_layer:
        mlp_param_optimizer = list(model.mlp.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in mlp_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in mlp_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},
        {'params': [p for n, p in cls_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in cls_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ] if args.use_ffnn_layer else [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},
        {'params': [p for n, p in cls_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in cls_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
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
    best_f1 = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model)
        micro_f1, macro_f1 = metrics['micro avg']['f1-score'], metrics['macro avg']['f1-score']
        dev_f1 = metrics['weighted avg']['f1-score']
        logger.info(f'Dev: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f} weighted_f1 - {(100*dev_f1):0.4f}')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_macrof1_{(100*macro_f1):0.4f}_microf1_{(100*micro_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, batch_size=1, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        micro_f1, macro_f1, dev_f1 = metrics['micro avg']['f1-score'], metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
        logger.info(f'Test: micro_F1 - {(100*micro_f1):0.4f} macro_f1 - {(100*macro_f1):0.4f} weighted_f1 - {(100*dev_f1):0.4f}')

def predict(args, sentence:str, model, tokenizer):
    inputs = tokenizer(
        sentence, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt", 
        return_offsets_mapping=True
    )
    offsets = inputs.pop('offset_mapping').squeeze(0)
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    predictions = model.crf.decode(logits, inputs['batch_inputs']['attention_mask'])
    predictions = predictions.squeeze(0)[0].cpu().numpy().tolist()

    pred_label = []
    idx = 1
    while idx < len(predictions) - 1:
        pred = predictions[idx]
        label = args.id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) - 1 and 
                args.id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            start, end = start.item(), end.item()
            word = sentence[start:end]
            pred_label.append({
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end
            })
        idx += 1
    return pred_label

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
    # Prepare task
    args.id2label = {0:'O'}
    for c in CATEGORIES:
        args.id2label[len(args.id2label)] = f"B-{c}"
        args.id2label[len(args.id2label)] = f"I-{c}"
    args.label2id = {v: k for k, v in args.id2label.items()}
    args.num_labels = len(args.id2label)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = BertCrfForNER.from_pretrained(
        args.model_checkpoint, 
        config=config, 
        args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = PeopleDaily(args.train_file)
        dev_dataset = PeopleDaily(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = PeopleDaily(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = PeopleDaily(args.test_file)
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            logger.info(f'predicting labels of {save_weight}...')
            
            results = []
            model.eval()
            for s_idx in tqdm(range(len(test_dataset))):
                sample = test_dataset[s_idx]
                pred_label = predict(args, sample['sentence'], model, tokenizer)
                results.append({
                    "sentence": sample['sentence'], 
                    "pred_label": pred_label, 
                    "true_label": sample['labels']
                })
            with open(os.path.join(args.output_dir, save_weight + '_test_data_pred.json'), 'wt', encoding='utf-8') as f:
                for exapmle_result in results:
                    f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')