import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
import json
import collections
import sys
from tqdm.auto import tqdm
sys.path.append('./')
from data.cmrc2018.cmrc2018_evaluate import evaluate

max_length = 384
stride = 128
n_best = 20
max_answer_length = 30
batch_size = 4
learning_rate = 1e-5
epoch_num = 3

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = CMRC2018('data/cmrc2018/cmrc2018_train.json')
valid_data = CMRC2018('data/cmrc2018/cmrc2018_dev.json')
test_data = CMRC2018('data/cmrc2018/cmrc2018_trial.json')

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return batch_data, torch.tensor(start_positions), torch.tensor(end_positions)

def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping').numpy().tolist()
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_mapping[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return batch_data, offset_mapping, example_ids

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)

print('train set size: ', )
print(len(train_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in train_dataloader]))
print('valid set size: ')
print(len(valid_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in valid_dataloader]))
print('test set size: ')
print(len(test_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in test_dataloader]))

class BertForExtractiveQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits

config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
model = BertForExtractiveQA.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, start_pos, end_pos) in enumerate(dataloader, start=1):
        X, start_pos, end_pos = X.to(device), start_pos.to(device), end_pos.to(device)
        start_pred, end_pred = model(X)
        start_loss = loss_fn(start_pred, start_pos)
        end_loss = loss_fn(end_pred, end_pos)
        loss = (start_loss + end_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, dataset, model):
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            pred_start_logits, pred_end_logit = model(batch_data)
        start_logits.append(pred_start_logits.cpu().numpy())
        end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]} for s_idx in range(len(dataset))
    ]
    predicted_answers = []
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
    return result

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model)
    avg_score = valid_scores['avg']
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
print("Done!")

# model.load_state_dict(torch.load('epoch_2_valid_avg_75.2441_model_weights.bin'))

# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     all_example_ids = []
#     all_offset_mapping = []
#     for _, offset_mapping, example_ids in test_dataloader:
#         all_example_ids += example_ids
#         all_offset_mapping += offset_mapping
#     example_to_features = collections.defaultdict(list)
#     for idx, feature_id in enumerate(all_example_ids):
#         example_to_features[feature_id].append(idx)

#     start_logits = []
#     end_logits = []
#     model.eval()
#     for batch_data, _, _ in tqdm(test_dataloader):
#         batch_data = batch_data.to(device)
#         pred_start_logits, pred_end_logit = model(batch_data)
#         start_logits.append(pred_start_logits.cpu().numpy())
#         end_logits.append(pred_end_logit.cpu().numpy())
#     start_logits = np.concatenate(start_logits)
#     end_logits = np.concatenate(end_logits)
    
#     theoretical_answers = [
#         {"id": test_data[s_idx]["id"], "answers": test_data[s_idx]["answers"]} for s_idx in range(len(test_dataloader))
#     ]
#     predicted_answers = []
#     save_resluts = []
#     for s_idx in tqdm(range(len(test_data))):
#         example_id = test_data[s_idx]["id"]
#         context = test_data[s_idx]["context"]
#         title = test_data[s_idx]["title"]
#         question = test_data[s_idx]["question"]
#         labels = test_data[s_idx]["answers"]
#         answers = []
#         # Loop through all features associated with that example
#         for feature_index in example_to_features[example_id]:
#             start_logit = start_logits[feature_index]
#             end_logit = end_logits[feature_index]
#             offsets = all_offset_mapping[feature_index]

#             start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
#             end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     if offsets[start_index] is None or offsets[end_index] is None:
#                         continue
#                     if (end_index < start_index or end_index-start_index+1 > max_answer_length):
#                         continue
#                     answers.append({
#                         "start": offsets[start_index][0], 
#                         "text": context[offsets[start_index][0] : offsets[end_index][1]], 
#                         "logit_score": start_logit[start_index] + end_logit[end_index],
#                     })
#         # Select the answer with the best score
#         if len(answers) > 0:
#             best_answer = max(answers, key=lambda x: x["logit_score"])
#             predicted_answers.append({
#                 "id": example_id, 
#                 "prediction_text": best_answer["text"], 
#                 "answer_start": best_answer["start"]
#             })
#             save_resluts.append({
#                 "id": example_id, 
#                 "title": title, 
#                 "context": context, 
#                 "question": question, 
#                 "answers": labels, 
#                 "prediction_text": best_answer["text"], 
#                 "answer_start": best_answer["start"]
#             })
#         else:
#             predicted_answers.append({
#                 "id": example_id, 
#                 "prediction_text": "", 
#                 "answer_start": 0
#             })
#             save_resluts.append({
#                 "id": example_id, 
#                 "title": title, 
#                 "context": context, 
#                 "question": question, 
#                 "answers": labels, 
#                 "prediction_text": "", 
#                 "answer_start": 0
#             })
#     eval_result = evaluate(predicted_answers, theoretical_answers)
#     print(f"F1: {eval_result['f1']:>0.2f} EM: {eval_result['em']:>0.2f} AVG: {eval_result['avg']:>0.2f}\n")
#     print('saving predicted results...')
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for example_result in save_resluts:
#             f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
