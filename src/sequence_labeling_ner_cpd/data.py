from torch.utils.data import Dataset, DataLoader
import numpy as np

CATEGORIES = ['LOC', 'ORG', 'PER']

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

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):
    
    def collote_fn(batch_samples):
        batch_sentence, batch_labels  = [], []
        for sample in batch_samples:
            batch_sentence.append(sample['sentence'])
            batch_labels.append(sample['labels'])
        batch_inputs = tokenizer(
            batch_sentence, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
        for s_idx, sentence in enumerate(batch_sentence):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            for char_start, char_end, _, tag in batch_labels[s_idx]:
                token_start = encoding.char_to_token(char_start)
                token_end = encoding.char_to_token(char_end)
                if not token_start or not token_end:
                    continue
                batch_label[s_idx][token_start] = args.label2id[f"B-{tag}"]
                batch_label[s_idx][token_start+1:token_end+1] = args.label2id[f"I-{tag}"]
        return {
            'batch_inputs': batch_inputs, 
            'labels': batch_label
        }
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)
