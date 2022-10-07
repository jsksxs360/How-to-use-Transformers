from torch.utils.data import Dataset, DataLoader
import torch

MAX_DATASET_SIZE = 200000

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= MAX_DATASET_SIZE:
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

def get_dataLoader(args, dataset, model, tokenizer, batch_size=None, shuffle=False):
    
    def collote_fn(batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            batch_inputs.append(sample['content'])
            batch_targets.append(sample['title'])
        batch_data = tokenizer(
            batch_inputs, 
            padding=True, 
            max_length=args.max_input_length,
            truncation=True, 
            return_tensors="pt"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch_targets, 
                padding=True, 
                max_length=args.max_target_length,
                truncation=True, 
                return_tensors="pt"
            )["input_ids"]
            batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            batch_data['labels'] = labels
        return batch_data
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn)
