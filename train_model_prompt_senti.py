import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers.activations import ACT2FN
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import json

max_length = 512
batch_size = 4
learning_rate = 1e-5
epoch_num = 3

prompt = lambda x: '总体上来说很[MASK]。' + x
pos_token, neg_token = '好', '差'

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(12)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class ChnSentiCorp(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2
                Data[idx] = {
                    'comment': items[0], 
                    'label': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = ChnSentiCorp('data/ChnSentiCorp/train.txt')
valid_data = ChnSentiCorp('data/ChnSentiCorp/dev.txt')
test_data = ChnSentiCorp('data/ChnSentiCorp/test.txt')

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pos_id = tokenizer.convert_tokens_to_ids(pos_token)
neg_id = tokenizer.convert_tokens_to_ids(neg_token)

def collote_fn(batch_samples):
    batch_sentence, batch_senti  = [], []
    for sample in batch_samples:
        batch_sentence.append(prompt(sample['comment']))
        batch_senti.append(sample['label'])
    batch_inputs = tokenizer(
        batch_sentence, 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.full(batch_inputs['input_ids'].shape, -100)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, max_length=max_length, truncation=True)
        mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
        batch_label[s_idx][mask_idx] = pos_id if batch_senti[s_idx] == '1' else neg_id
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertForPrompt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        prediction_scores = self.cls(sequence_output)
        return prediction_scores

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        predictions = model(X)
        loss = loss_fn(predictions.view(-1, config.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, dataset, model):
    results = []
    model.eval()
    for batch_data, _ in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            token_logits = model(batch_data)
        mask_token_indexs = torch.where(batch_data["input_ids"] == tokenizer.mask_token_id)[1]
        for s_idx, mask_idx in enumerate(mask_token_indexs):
            results.append(token_logits[s_idx, mask_idx, [neg_id, pos_id]].cpu().numpy())
    true_labels = [
        int(dataset[s_idx]['label']) for s_idx in range(len(dataset))
    ]
    predictions = np.asarray(results).argmax(axis=-1).tolist()
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f}, neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
    print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
    return metrics

# loss_fn = nn.CrossEntropyLoss()
# optimizer = AdamW(model.parameters(), lr=learning_rate)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=epoch_num*len(train_dataloader),
# )

# total_loss = 0.
# best_f1_score = 0.
# for t in range(epoch_num):
#     print(f"Epoch {t+1}/{epoch_num}\n" + 30 * "-")
#     total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
#     valid_scores = test_loop(valid_dataloader, valid_data, model)
#     macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
#     f1_score = (macro_f1 + micro_f1) / 2
#     if f1_score > best_f1_score:
#         best_f1_score = f1_score
#         print('saving new weights...\n')
#         torch.save(
#             model.state_dict(), 
#             f'epoch_{t+1}_valid_macrof1_{(macro_f1*100):0.3f}_microf1_{(micro_f1*100):0.3f}_model_weights.bin'
#         )
# print("Done!")

model.load_state_dict(torch.load('epoch_3_valid_macrof1_95.331_microf1_95.333_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    results = []
    for batch_data, _ in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            token_logits = model(batch_data)
        mask_token_indexs = torch.where(batch_data["input_ids"] == tokenizer.mask_token_id)[1]
        for s_idx, mask_idx in enumerate(mask_token_indexs):
            results.append(token_logits[s_idx, mask_idx, [neg_id, pos_id]].cpu().numpy())
    true_labels = [
        int(test_data[s_idx]['label']) for s_idx in range(len(test_data))
    ]
    predictions = np.asarray(results).argmax(axis=-1).tolist()
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        comment, label = test_data[s_idx]['comment'], test_data[s_idx]['label']
        probs = torch.nn.functional.softmax(torch.tensor(results[s_idx]), dim=-1)
        save_resluts.append({
            "comment": comment, 
            "label": label, 
            "pred": '1' if probs[1] > probs[0] else '0', 
            "prediction": {'0': probs[0].item(), '1': probs[1].item()}
        })
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f}, neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
    print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
    print('saving predicted results...')
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in save_resluts:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
