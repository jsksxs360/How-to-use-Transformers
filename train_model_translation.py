import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm
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
seed_everything(42)

max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

max_length = 128

batch_size = 32
learning_rate = 1e-5
epoch_num = 3

class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = TRANS('data/translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('data/translation2019zh/translation2019zh_valid.json')

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    batch_data = tokenizer(
        batch_inputs, 
        text_target=batch_targets, 
        padding=True, 
        max_length=max_length,
        truncation=True, 
        return_tensors="pt"
    )
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data['labels'])
    end_token_index = torch.where(batch_data['labels'] == tokenizer.eos_token_id)[1]
    for idx, end_idx in enumerate(end_token_index):
        batch_data['labels'][idx][end_idx+1:] = -100
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

bleu = BLEU()

def test_loop(dataloader, model):
    preds, labels = [], []
    
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_length,
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    return bleu.corpus_score(preds, labels).score

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_bleu = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_bleu = test_loop(valid_dataloader, model)
    print(f"BLEU: {valid_bleu:>0.2f}\n")
    if valid_bleu > best_bleu:
        best_bleu = valid_bleu
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{t+1}_valid_bleu_{valid_bleu:0.2f}_model_weights.bin'
        )
print("Done!")

# import json

# model.load_state_dict(torch.load('epoch_1_valid_bleu_53.38_model_weights.bin'))

# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     sources, preds, labels = [], [], []
#     for batch_data in tqdm(test_dataloader):
#         batch_data = batch_data.to(device)
#         generated_tokens = model.generate(
#             batch_data["input_ids"],
#             attention_mask=batch_data["attention_mask"],
#             max_length=max_length,
#         ).cpu().numpy()
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
#         labels += [[label.strip()] for label in decoded_labels]
#     bleu_score = bleu.corpus_score(preds, labels).score
#     print(f"Test BLEU: {bleu_score:>0.2f}\n")
#     results = []
#     print('saving predicted results...')
#     for source, pred, label in zip(sources, preds, labels):
#         results.append({
#             "sentence": source, 
#             "prediction": pred, 
#             "translation": label[0]
#         })
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for exapmle_result in results:
#             f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
