from torch.utils.data import Dataset, DataLoader

def get_prompt(x):
    prompt = f'总体上来说很[MASK]。{x}'
    return {
        'prompt': prompt, 
        'mask_offset': prompt.find('[MASK]')
    }

def get_verbalizer(tokenizer, vtype):
    assert vtype in ['base', 'virtual']
    return {
        'pos': {'token': '好', 'id': tokenizer.convert_tokens_to_ids("好")}, 
        'neg': {'token': '差', 'id': tokenizer.convert_tokens_to_ids("差")}
    } if vtype == 'base' else {
        'pos': {
            'token': '[POS]', 'id': tokenizer.convert_tokens_to_ids("[POS]"), 
            'description': '好的、优秀的、正面的评价、积极的态度'
        }, 
        'neg': {
            'token': '[NEG]', 'id': tokenizer.convert_tokens_to_ids("[NEG]"), 
            'description': '差的、糟糕的、负面的评价、消极的态度'
        }
    }

class ChnSentiCorp(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('\t')
                assert len(items) == 2
                prompt_data = get_prompt(items[0])
                Data[idx] = {
                    'comment': items[0], 
                    'prompt': prompt_data['prompt'], 
                    'mask_offset': prompt_data['mask_offset'], 
                    'label': items[1]
                }
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, verbalizer, batch_size=None, shuffle=False):
    pos_id, neg_id = verbalizer['pos']['id'], verbalizer['neg']['id']
    
    def collote_fn(batch_samples):
        batch_sentences, batch_mask_idxs, batch_labels  = [], [], []
        for sample in batch_samples:
            batch_sentences.append(sample['prompt'])
            encoding = tokenizer(sample['prompt'], truncation=True)
            mask_idx = encoding.char_to_token(sample['mask_offset'])
            assert mask_idx is not None
            batch_mask_idxs.append(mask_idx)
            batch_labels.append(int(sample['label']))
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        label_word_id = [neg_id, pos_id]
        return {
            'batch_inputs': batch_inputs, 
            'batch_mask_idxs': batch_mask_idxs, 
            'label_word_id': label_word_id, 
            'labels': batch_labels
        }
    
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=collote_fn
    )
