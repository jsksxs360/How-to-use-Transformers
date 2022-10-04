import os
import numpy as np
import random
from tqdm.auto import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed_everything(7)

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

categories = set()

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        labels.append([i, i, char, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        labels[-1][1] = i
                        labels[-1][2] += char
                Data[idx] = {
                    'sentence': sentence, 
                    'labels': labels
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = PeopleDaily('data/china-people-daily-ner-corpus/example.train')
valid_data = PeopleDaily('data/china-people-daily-ner-corpus/example.dev')
test_data = PeopleDaily('data/china-people-daily-ner-corpus/example.test')

id2label = {0:'O'}
for c in list(sorted(categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}

def collote_fn(batch_samples):
    batch_sentence, batch_labels  = [], []
    for sample in batch_samples:
        batch_sentence.append(sample['sentence'])
        batch_labels.append(sample['labels'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        batch_label[s_idx][0] = -100
        batch_label[s_idx][len(encoding.tokens())-1:] = -100
        for char_start, char_end, _, tag in batch_labels[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start+1:token_end+1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, len(id2label))
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
model = BertForNER.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0, 2, 1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    true_labels, true_predictions = [], []

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()
            true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    return classification_report(
      true_labels, 
      true_predictions, 
      mode='strict', 
      scheme=IOB2, 
      output_dict=True
    )

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_f1 = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    metrics = test_loop(valid_dataloader, model)
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{t+1}_valid_macrof1_{(100*valid_macro_f1):0.3f}_microf1_{(100*valid_micro_f1):0.3f}_weights.bin'
        )
print("Done!")

# import json

# model.load_state_dict(
#     torch.load('epoch_3_valid_macrof1_95.878_microf1_96.049_weights.bin', map_location=torch.device('cpu'))
# )
# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     true_labels, true_predictions = [], []
#     for X, y in tqdm(test_dataloader):
#         X, y = X.to(device), y.to(device)
#         pred = model(X)
#         predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
#         labels = y.cpu().numpy().tolist()
#         true_labels += [[id2label[int(l)] for l in label if l != -100] for label in labels]
#         true_predictions += [
#             [id2label[int(p)] for (p, l) in zip(prediction, label) if l != -100]
#             for prediction, label in zip(predictions, labels)
#         ]
#     print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
#     results = []
#     print('predicting labels...')
#     for s_idx in tqdm(range(len(test_data))):
#         example = test_data[s_idx]
#         inputs = tokenizer(example['sentence'], truncation=True, return_tensors="pt", return_offsets_mapping=True)
#         offsets = inputs.pop('offset_mapping').squeeze(0)
#         inputs = inputs.to(device)
#         pred = model(inputs)
#         probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
#         predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()

#         pred_label = []
#         idx = 0
#         while idx < len(predictions):
#             pred = predictions[idx]
#             label = id2label[pred]
#             if label != "O":
#                 label = label[2:] # Remove the B- or I-
#                 start, end = offsets[idx]
#                 all_scores = [probabilities[idx][pred]]
#                 # Grab all the tokens labeled with I-label
#                 while (
#                     idx + 1 < len(predictions) and 
#                     id2label[predictions[idx + 1]] == f"I-{label}"
#                 ):
#                     all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
#                     _, end = offsets[idx + 1]
#                     idx += 1

#                 score = np.mean(all_scores).item()
#                 start, end = start.item(), end.item()
#                 word = example['sentence'][start:end]
#                 pred_label.append(
#                     {
#                         "entity_group": label,
#                         "score": score,
#                         "word": word,
#                         "start": start,
#                         "end": end,
#                     }
#                 )
#             idx += 1
#         results.append(
#             {
#                 "sentence": example['sentence'], 
#                 "pred_label": pred_label, 
#                 "true_label": example['labels']
#             }
#         )
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for exapmle_result in results:
#             f.write(json.dumps(exapmle_result, ensure_ascii=False) + '\n')
