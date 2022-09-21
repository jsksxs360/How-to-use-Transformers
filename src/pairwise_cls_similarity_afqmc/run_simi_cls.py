import os
import logging
import json
import torch
from transformers import AdamW, get_scheduler
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append('../../')
from src.tools import seed_everything
from src.pairwise_cls_similarity_afqmc.arg import parse_args
from src.pairwise_cls_similarity_afqmc.data import AFQMC, get_dataLoader
from src.pairwise_cls_similarity_afqmc.modeling import BertForPairwiseCLS, RobertaForPairwiseCLS
from tqdm.auto import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

MODEL_CLASSES = {
    'bert': BertForPairwiseCLS, 
    # 'roberta': RobertaForPairwiseCLS, 
    'roberta': BertForPairwiseCLS # 哈工大权重必须使用Bert模型加载
}

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

def test_loop(args, dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]

            predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = batch_data['labels'].cpu().numpy()
            correct += (predictions == labels).sum()

    correct /= len(dataloader.dataset)
    return correct

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
    best_acc = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        valid_acc = test_loop(args, dev_dataloader, model)
        logger.info(f"Dev Accuracy: {(100*valid_acc):>0.2f}%")
        if valid_acc > best_acc:
            best_acc = valid_acc
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_acc_{(100*valid_acc):0.1f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        test_acc = test_loop(args, test_dataloader, model)
        logger.info(f"Test Accuracy: {(100*test_acc):>0.2f}%")

def predict(args, sent_1, sent_2, model, tokenizer):
    
    inputs = tokenizer(
        sent_1, 
        sent_2, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    pred = int(logits.argmax(dim=-1)[0].cpu().numpy())
    prob = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    return pred, prob[pred]

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
    args.num_labels = 2
    model = MODEL_CLASSES[args.model_type].from_pretrained(
        args.model_checkpoint, 
        config=config, 
        args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = AFQMC(args.train_file)
        dev_dataset = AFQMC(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = AFQMC(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = []
        with open(args.test_file, 'rt') as f:
            for idx, line in enumerate(f):
                test_dataset.append(json.loads(line.strip()))
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
            logger.info(f'predicting labels of {save_weight}...')

            results = []
            model.eval()
            for sample in tqdm(test_dataset):
                pred, prob = predict(args, sample['sentence1'], sample['sentence2'], model, tokenizer)
                results.append({
                    "sentence1": sample['sentence1'], 
                    "sentence2": sample['sentence2'], 
                    "label": sample['label'],
                    "pred_label": str(pred), 
                    "pred_prob": prob
                })
            with open(os.path.join(args.output_dir, save_weight + '_test_pred_simi.json'), 'wt', encoding='utf-8') as f:
                for sample_result in results:
                    f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
