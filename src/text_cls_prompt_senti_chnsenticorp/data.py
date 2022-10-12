from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

PROMPT = lambda x: '总体上来说很[MASK]。' + x
POS_TOKEN, NEG_TOKEN = '好', '差'

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

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False):

    pos_id = tokenizer.convert_tokens_to_ids(POS_TOKEN)
    neg_id = tokenizer.convert_tokens_to_ids(NEG_TOKEN)
    
    def collote_fn(batch_samples):
        batch_sentence, batch_senti  = [], []
        for sample in batch_samples:
            batch_sentence.append(PROMPT(sample['comment']))
            batch_senti.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sentence, 
            max_length=args.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_label = np.full(batch_inputs['input_ids'].shape, -100)
        for s_idx, sentence in enumerate(batch_sentence):
            encoding = tokenizer(sentence, max_length=args.max_length, truncation=True)
            mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
            batch_label[s_idx][mask_idx] = pos_id if batch_senti[s_idx] == '1' else neg_id
        return {
            'batch_inputs': batch_inputs, 
            'labels': batch_label
        }
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)
