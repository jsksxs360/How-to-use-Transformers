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
    batch_sentences, batch_labels  = [], []
    for sample in batch_samples:
        batch_sentences.append(prompt(sample['comment']))
        batch_labels.append(int(sample['label']))
    batch_inputs = tokenizer(
        batch_sentences, 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_mask_idx, label_word_id = [], [neg_id, pos_id]
    for sentence in batch_sentences:
        encoding = tokenizer(sentence, truncation=True)
        mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
        batch_mask_idx.append(mask_idx)
    return batch_inputs, torch.tensor(batch_mask_idx), label_word_id, torch.tensor(batch_labels)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

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
    
    def forward(self, batch_x, batch_mask_idx, label_word_id):
        bert_output = self.bert(**batch_x)
        sequence_output = bert_output.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idx.unsqueeze(-1)).squeeze(1)
        prediction_scores = self.cls(batch_mask_reps)
        return prediction_scores[:, label_word_id]

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (batch_X, batch_mask_idx, label_word_id, batch_y) in enumerate(dataloader, start=1):
        batch_X, batch_mask_idx, batch_y = batch_X.to(device), batch_mask_idx.to(device), batch_y.to(device)
        predictions = model(batch_X, batch_mask_idx, label_word_id)
        loss = loss_fn(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_mask_idx, label_word_id, batch_y in dataloader:
            true_labels += batch_y.numpy().tolist()
            batch_X, batch_mask_idx = batch_X.to(device), batch_mask_idx.to(device)
            pred = model(batch_X, batch_mask_idx, label_word_id)
            predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f}, neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
    print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
    return metrics

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_f1_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n" + 30 * "-")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, model)
    macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
    f1_score = (macro_f1 + micro_f1) / 2
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{t+1}_valid_macrof1_{(macro_f1*100):0.3f}_microf1_{(micro_f1*100):0.3f}_model_weights.bin'
        )
print("Done!")

# model.load_state_dict(torch.load('epoch_3_valid_macrof1_94.748_microf1_94.749_model_weights.bin'))

# model.eval()
# with torch.no_grad():
#     print('evaluating on test set...')
#     true_labels, predictions, probs = [], [], []
#     for batch_X, batch_mask_idx, label_word_id, batch_y in tqdm(test_dataloader):
#         true_labels += batch_y.numpy().tolist()
#         batch_X, batch_mask_idx = batch_X.to(device), batch_mask_idx.to(device)
#         pred = model(batch_X, batch_mask_idx, label_word_id)
#         predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
#         probs += torch.nn.functional.softmax(pred, dim=-1)
#     save_resluts = []
#     for s_idx in tqdm(range(len(test_data))):
#         save_resluts.append({
#             "comment": test_data[s_idx]['comment'], 
#             "label": true_labels[s_idx], 
#             "pred": predictions[s_idx], 
#             "prob": {'neg': probs[s_idx][0].item(), 'pos': probs[s_idx][1].item()}
#         })
#     metrics = classification_report(true_labels, predictions, output_dict=True)
#     pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
#     neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
#     macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
#     print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f}, neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
#     print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
#     print('saving predicted results...')
#     with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
#         for example_result in save_resluts:
#             f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
