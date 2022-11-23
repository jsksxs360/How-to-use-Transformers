import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from rouge import Rouge
import random
import numpy as np
import os
import json

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed_everything(5)

max_dataset_size = 200000
max_input_length = 512
max_target_length = 32

batch_size = 32
learning_rate = 1e-5
epoch_num = 3

beam_size = 4
no_repeat_ngram_size = 2

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('data/lcsts_tsv/data1.tsv')
valid_data = LCSTS('data/lcsts_tsv/data2.tsv')
test_data = LCSTS('data/lcsts_tsv/data3.tsv')

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs, 
        padding=True, 
        max_length=max_input_length,
        truncation=True, 
        return_tensors="pt"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets, 
            padding=True, 
            max_length=max_target_length,
            truncation=True, 
            return_tensors="pt"
        )["input_ids"]
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
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

rouge = Rouge()

def test_loop(dataloader, model):
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"Rouge1: {result['rouge-1']:>0.2f} Rouge2: {result['rouge-2']:>0.2f} RougeL: {result['rouge-l']:>0.2f}\n")
    return result

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_rouge = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model)
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
print("Done!")

# model.load_state_dict(torch.load('epoch_1_valid_rouge_6.6667_model_weights.bin'))

# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     sources, preds, labels = [], [], []
#     for batch_data in tqdm(test_dataloader):
#         batch_data = batch_data.to(device)
#         generated_tokens = model.generate(
#             batch_data["input_ids"],
#             attention_mask=batch_data["attention_mask"],
#             max_length=max_target_length,
#             num_beams=beam_size,
#             no_repeat_ngram_size=no_repeat_ngram_size,
#         ).cpu().numpy()
#         if isinstance(generated_tokens, tuple):
#             generated_tokens = generated_tokens[0]
#         label_tokens = batch_data["labels"].cpu().numpy()

#         decoded_sources = tokenizer.batch_decode(
#             batch_data["input_ids"].cpu().numpy(), 
#             skip_special_tokens=True, 
#             use_source_tokenizer=True
#         )
#         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#         label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
#         decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

#         sources += [source.strip() for source in decoded_sources]
#         preds += [pred.strip() for pred in decoded_preds]
#         labels += [label.strip() for label in decoded_labels]
#     scores = rouge.get_scores(
#         hyps=[' '.join(pred) for pred in preds], 
#         refs=[' '.join(label) for label in labels], 
#         avg=True
#     )
#     rouges = {key: value['f'] * 100 for key, value in scores.items()}
#     rouges['avg'] = np.mean(list(rouges.values()))
#     print(f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
#     results = []
#     print('saving predicted results...')
#     for source, pred, label in zip(sources, preds, labels):
#         results.append({
#             "document": source, 
#             "prediction": pred, 
#             "summarization": label
#         })
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for exapmle_result in results:
#             f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
