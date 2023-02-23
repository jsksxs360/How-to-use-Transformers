---
title: 第十三章：Prompt 情感分析
author: SHENG XU
date: 2022-10-10
category: NLP
layout: post
---

本文我们将运用 Transformers 库来完成情感分析任务，并且使用目前流行的 Prompt 方法。Prompt 方法的核心想法就是使用模板将问题转换为模型预训练任务类似的形式来处理。

例如要判断标题“American Duo Wins Opening Beach Volleyball Match”的新闻类别，就可以应用模板“This is a $\texttt{[MASK]}$ News: $\textbf{x}$”将其转换为“This is a $\texttt{[MASK]}$ News: American Duo Wins Opening Beach Volleyball Match”，然后送入到经过 MLM (Mask Language Modeling) 任务预训练的模型中预测 $\texttt{[MASK]}$ 对应的词，最后将词映射到新闻类别（比如“Sports”对应“体育”类）。

> 如果你对 Prompt 方法不是很熟悉，建议可以先阅读一下[《Prompt 方法简介》](/2022/09/10/what-is-prompt.html)

下面我们以情感分析任务为例，运用 Transformers 库手工构建一个 Prompt 模型来完成任务。

## 1. 准备数据

这里我们选择中文情感分析语料库 ChnSentiCorp 作为数据集，该语料基于爬取的酒店、电脑、书籍网购评论构建，共包含评论接近一万条，可以从百度 ERNIE [示例仓库](https://ernie-github.cdn.bcebos.com/data-chnsenticorp.tar.gz)或者[百度云盘](https://pan.baidu.com/s/18UROBO8t1bDGn_spWnXg4w?pwd=xszb)下载。

语料库已经划分好了训练集、验证集和测试集，分别包含 9600 / 1200 / 1200 条评论。一行是一个样本，使用 `TAB` 分隔评论和对应的标签（0-消极，1-积极）：

```
选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般	1
...
```

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签。

```python
from torch.utils.data import Dataset

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
```

下面我们输出数据集的尺寸，并且打印出一个训练样本：

```python
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))
```

```
train set size: 9600
valid set size: 1200
test set size: 1200
{'comment': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 'label': '1'}
```

### 数据预处理

接下来我们就需要通过 `DataLoader` 库来按 batch 加载数据，将文本转换为模型可以接受的 token IDs。

大部分 Prompt 方法都是通过模板将问题转换为 MLM 任务的形式来解决，同样地，这里我们定义模板为“总体上来说很 $\texttt{[MASK]}$。$\text{x}$”，并且规定如果 $\texttt{[MASK]}$ 预测为 token “好”就判定情感为“积极”，如果预测为 token “差”就判定为“消极”。

MLM 任务与[序列标注](/2022/03/18/transformers-note-6.html)任务很相似，也是对 token 进行分类，并且类别是整个词表，不同之处在于 MLM 任务只需要对文中特殊的 $\texttt{[MASK]}$ token 进行标注。因此在处理数据时，我们还需要：1）记录下所有 $\texttt{[MASK]}$ token 的索引，以便在模型的输出序列中将它们的表示取出。2）记录下标签 token（例如这里的“好”和“差”）对应的 ID，因为我们实际上只关心模型对这些词语的预测结果。

下面以处理第一个样本为例。我们通过 `char_to_token()` 将 $\texttt{[MASK]}$ 从原文位置映射到切分后的 token 索引，同时通过 `convert_tokens_to_ids` 来获取标签词“好”或“差”的 token ID。注意，由于标签 0 表示消极，1 表示积极，所以标签词的 token ID 也按照该顺序进行组织，以便后续从预测结果中直接取出它们对应的 logits 值。

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pos_id = tokenizer.convert_tokens_to_ids("好")
neg_id = tokenizer.convert_tokens_to_ids("差")
print(f'pos_id:{pos_id}\tneg_id:{neg_id}')

pre_text = '总体上来说很[MASK]。'
comment = '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
label = '1'

sentence = pre_text + comment
encoding = tokenizer(sentence, truncation=True)
tokens = encoding.tokens()
mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
label_word_ids = [neg_id, pos_id]

print(tokens)
print('mask_idx:', mask_idx)
print('label_word_ids:', label_word_ids)
```

```
pos_id:1962     neg_id:2345
['[CLS]', '总', '体', '上', '来', '说', '很', '[MASK]', '。', '这', '个', '宾', '馆', '比', '较', '陈', '旧', '了', '，', '特', '价', '的', '房', '间', '也', '很', '一', '般', '。', '总', '体', '来', '说', '一', '般', '。', '[SEP]']
mask_idx: 7
label_word_ids: [2345, 1962]
```

可以看到，BERT 分词器正确地将“[MASK]”识别为一个 token，并且成功记录下 $\texttt{[MASK]}$ token 在序列中的索引以及标签词“好”和“差”的 token ID。

> 注意，这里演示的是只对一个 $\texttt{[MASK]}$ token 进行预测的情况。如果模板中包含多个 $\texttt{[MASK]}$ token，并且每个 $\texttt{[MASK]}$ token 对应的标签词都不同，那么同样需要将这些信息都记录下来。

在实际编写 DataLoader 的批处理函数 `collate_fn()` 时，我们处理的不是一个而是多个样本，因此需要对上面的操作进行扩展。

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pre_text = '总体上来说很[MASK]。'
pos_id = tokenizer.convert_tokens_to_ids("好")
neg_id = tokenizer.convert_tokens_to_ids("差")

def collote_fn(batch_samples):
    batch_sentences, batch_labels  = [], []
    for sample in batch_samples:
        batch_sentences.append(pre_text + sample['comment'])
        batch_labels.append(int(sample['label']))
    batch_inputs = tokenizer(
        batch_sentences, 
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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_X, batch_mask_idx, label_word_id, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print(batch_X)
print(batch_mask_idx)
print(batch_y)
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 191]), 
    'token_type_ids': torch.Size([4, 191]), 
    'attention_mask': torch.Size([4, 191])
}
{'input_ids': tensor([
        [  101,  2600,   860,   677,  3341,  6432,  2523,   103,   511,  1762,
          5307,  1325,  6814,   749,  3739,  1366,  2161,  7667,  1469,  6205,
          3862,  7649,  2421,  4638,  2521,  6878,  1400,  8024,  1744,  5549,
          3187,  4542,  3221,   671,   702,  2661,  1599,   511,  2791,  7313,
          2160,  3139,  3146,  3815,  8024,  6629,  4772,  5543,   924,  6395,
          4717,   671,   702,  1962,  6230,   511,  3123,  1762,  1071,   800,
          1814,  2356,  8024,  1377,  5543,  1744,  5549,  6820,  1916,   679,
           677,  1724,  3215,  5277,  8024,   852,  1963,  3362,  6205,  3862,
          7649,  2421,   722,  3837,  6963,  5543,  5050,  1724,  3215,  5277,
          4638,  6413,  8024,  1744,  5549,  4696,  4638,  5050,  3221,  2595,
           817,  3683,  6631,  7770,   749,   511,  1765,  4415,   855,  5390,
           738,  1962,  8024,   817,  7178,   738,   912,  2139,  8024,  3221,
          3634,  3613,  7942,  2255,   722,  6121,  3297,  4007,  2692,  4638,
          6983,  2421,   511,   102,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0],
        [  101,  2600,   860,   677,  3341,  6432,  2523,   103,   511,  3209,
          3209,   743,   749,   127,  3315,   741,  8024,  1372,  1168,   749,
           124,  3315,  8024,   738,  3766,  3300,  6432,  3221,   784,   720,
          1333,  1728,  8024,   809,  1400,  2582,   720,   928,  4638,  6814,
          8043,  8043,  8043,  8043,  8043,  8043,  8043,  8043,  8043,  8043,
          8043,   102,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0],
        [  101,  2600,   860,   677,  3341,  6432,  2523,   103,   511,   817,
          3419,  1962,  1557,  8024,  3209,  3030,  4708,  4638,   511,  6929,
           763,  6163,   679,   677,  8766,  4638,  4692,  4692,  2769,  6432,
          4638,   511,  1168,  9324,   934,  3121, 10785,  9702,  1168, 11319,
          4197,  1400,  2823,   671,  2476,  2128,  6163,  4276,  3315,  4638,
          8766,  8024,  6822,  1343,  4684,  2970, 10843,  2957,  2792,  3300,
          1146,  1277,  4197,  1400,  7028,  3173,  1146,  1277,  2128,  6163,
          1315,  1377,   511,  2823,   671,  2476, 10797,  4638,  8766,  8024,
          6822,  1343,   886,  4500, 10797,  4638,  1146,  1277,  1216,  5543,
          4684,  2970,  1146,  1277,  2128,  6163,  1315,  1377,   511,  1963,
          3362,   872,  4801,  3221,  6206, 10785,  9702,  4638,  6413,  8024,
          1343,  2823,   671,   702, 10785,  9702,   118, 11319,  4638,  7721,
          1220,  8024,  2128,  6163,  1962,  1400,  1086,  2128,  6163,  6821,
           702,  7721,  1220,  2218,  1377,   809,  6760,  2940,  1168, 10785,
          9702,  1343,   511,   102,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0],
        [  101,  2600,   860,   677,  3341,  6432,  2523,   103,   511,   671,
          4684,  1599,  3614,  6929,   702,  2399,   807,  4638,   782,  4289,
          3125,   752,  2600,   833,  3300,  6929,   763,   679,  1728,  4535,
          2501,  4384,  1862,  5445,  3121,  1359,  4638,  5283,  5679,  3315,
          2595,  6929,  3416,  4638,  4696,  2658,  1963,   791,  4692,  3341,
          4994,  3221,  2208,   722,  1348,  2208,  1728,  1327,  2829,  5445,
          4397,  6586,  6821,  3416,  4638,  4263,  2658,  2215,  1071,  3918,
          1331,  1265,  6237,   679,  2458,  4638,  3849,  4164,  2697,  6375,
           782,  3617,  5387,   679,  5543,  7474,  4904,  2208,  6387,  4638,
          2697,  2595,   704,  2881,  3300,  3291,  1914,  1728,  4836,  7410,
          5445,  4495,  1139,  4638,  4415,  2595,  6821,  3416,  4638,  1957,
          2094,  1780,  2137,  3300,   928,  2573,  1762,  2496,  3198,  4638,
          4852,   833,  7027,  2418,  6421,  2400,   679,  2208,  6224,  1377,
          3221,  2769,   679,  4761,  6887,  1008,  5439,   676,  6821,  3416,
          4638,  4511,   782,  4590,  2552,  3300,  2857,  2496,  1294,  2209,
           679,  1127,  1348,  1398,  3416,  4638,  1780,  2137,  6821,   702,
           686,  4518,   677,  6820,   833,  3300,  1126,   702,  8043,   671,
           702,  1957,   782,  4638,   671,  4495,  7027,  2881,  3300,  6814,
          6821,  3416,  4638,  4511,  2094,  1923,  1908,   862,  3724,  8043,
           102]]), 
 'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
tensor([7, 7, 7, 7])
tensor([1, 0, 1, 1])
```

可以看到，DataLoader 按照我们设置的 batch size 每次对 4 个样本进行编码，将 token 序列填充到了相同的长度。这里由于我们对所有样本都添加相同的“前缀”，因此 `[MASK]` token 的索引都为 7。

## 2. 训练模型

### 构建模型

对于 MLM 任务，可以直接使用 Transformers 库封装好的 `AutoModelForMaskedLM` 类。由于 BERT 已经在 MLM 任务上进行了预训练，因此借助模板我们甚至可以在不微调的情况下 (Zero-shot) 直接使用模型来预测情感极性。例如对我们的第一个样本：

```python
import torch
from transformers import AutoModelForMaskedLM

checkpoint = "bert-base-chinese"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

text = "总体上来说很[MASK]。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。"
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
```

```
'>>> 总体上来说很好。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
'>>> 总体上来说很棒。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
'>>> 总体上来说很差。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
'>>> 总体上来说很般。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
'>>> 总体上来说很赞。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'
```

可以看到，BERT 模型成功地将 `[MASK]` token 预测成了我们预期的表意词“好”。这里我们还打印出了其他几个大概率的预测词，大部分都具有积极的情感（“好”、“棒”、“赞”）。

当然，这种方式不够灵活，因此像之前章节中一样，本文采用继承 Transformers 库预训练模型的方式来手工构建模型：

```python
from torch import nn
from transformers.activations import ACT2FN
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
config = AutoConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint, config=config).to(device)
print(model)
```

```
Using cpu device
BertForPrompt(
  (bert): BertModel()
  (cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=768, out_features=21128, bias=True)
    )
  )
)
```

注意，这里为了能够加载预训练好的 MLM head 参数，我们按照 Transformers 库中的模型结构来构建 `BertForPrompt` 模型。可以看到，BERT 自带的 MLM head 由两个部分组成：首先对所有 token 进行一个 $768 \times 768$ 的非线性映射（包括激活函数和 LayerNorm），然后使用一个 $768\times 21128$ 的线性映射预测词表中每个 token 的分数。

为了让模型适配我们的任务，这里首先通过 `batched_index_select` 函数从 BERT 的输出序列中抽取出 $\texttt{[MASK]}$ token 对应的表示，在运用 MLM head 预测出该 $\texttt{[MASK]}$ token 对应词表中每个 token 的分数之后，我们只返回“差”和“好”这两个标签词的分数用于分类。

为了测试模型的操作是否符合预期，我们尝试将一个 batch 的数据送入模型：

```python
batch_X, batch_mask_idx, label_word_id, batch_y = next(iter(train_dataloader))
outputs = model(batch_X, batch_mask_idx, label_word_id)
print(outputs.shape)
```

```
torch.Size([4, 2])
```

模型对每个样本都应该输出“差”和“好”这两个标签词的预测 logits 值（分别对应“消极”和“积极”两个类别），因此这里模型的输出尺寸 $4×2$  完全符合预期。

### 优化模型参数

与之前一样，我们将每一轮 Epoch 分为“训练循环”和“验证/测试循环”，在训练循环中计算损失、优化模型参数，在验证/测试循环中评估模型性能。下面我们首先实现训练循环。

因为对标签词的预测实际上就是对类别的预测，所以这里模型的输出与[同义句判断任务](https://transformers.run/intro/2021-12-17-transformers-note-4/)中介绍过的普通文本分类模型完全一致，损失也同样是通过在类别预测和答案标签之间计算交叉熵：

```python
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
```

验证/测试循环负责评估模型的性能。对于分类任务最常见的就是通过精确率、召回率、F1值 (P / R / F1) 指标来评估每个类别的预测性能，然后再通过宏/微 F1 值 (Macro-F1/Micro-F1) 来评估整体分类性能。

这里我们借助机器学习包 [sklearn](https://scikit-learn.org/stable/#) 提供的 `classification_report` 函数来输出这些指标，例如：

```python
from sklearn.metrics import classification_report

y_true = [1, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0, 1, 0]

print(classification_report(y_true, y_pred, output_dict=False))
```

```
              precision    recall  f1-score   support

           0       0.50      0.75      0.60         4
           1       0.67      0.57      0.62         7
           2       1.00      0.50      0.67         2

    accuracy                           0.62        13
   macro avg       0.72      0.61      0.63        13
weighted avg       0.67      0.62      0.62        13
```

因此在验证/测试循环中，我们只需要汇总模型对所有样本的预测结果和答案标签，然后送入到 `classification_report` 中计算各项分类指标：

```python
from sklearn.metrics import classification_report

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
```

为了方便后续保存验证集上最好的模型，这里我们还返回了评估结果。

### 保存模型

与之前一样，我们会根据模型在验证集上的性能来调整超参数以及选出最好的模型权重，然后将选出的模型应用于测试集以评估最终的性能。这里我们继续使用 AdamW 优化器，并且通过 `get_scheduler()` 函数定义学习率调度器：

```python
from transformers import AdamW, get_scheduler

learning_rate = 1e-5
epoch_num = 3

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
```

在开始训练之前，我们先评估一下没有微调的 BERT 模型在测试集上的性能。

```python
test_data = ChnSentiCorp('data/ChnSentiCorp/test.txt')
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

test_loop(test_dataloader, model)
```

```
100%|█████████████████████████████████| 300/300 [00:06<00:00, 44.78it/s]
pos: 53.05 / 100.00 / 69.33, neg: 100.00 / 9.12 / 16.72
Macro-F1: 43.02 Micro-F1: 43.37
```

可以看到，得益于 Prompt 方法，不经微调的 BERT 模型也已经具有初步的情感分析能力，在测试集上的 Macro-F1 和 Micro-F1 值分别为 43.02 和 43.37。有趣的是，“积极”类别的召回率和“消极”类别的准确率都为 100%，这说明 BERT 对大部分样本都倾向于判断为“积极”类（可能预训练时看到的积极性文本更多吧）。

下面，我们正式开始训练，完整的训练代码如下：

```python
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

pos_id = tokenizer.convert_tokens_to_ids("好")
neg_id = tokenizer.convert_tokens_to_ids("差")

def collote_fn(batch_samples):
    batch_sentences, batch_labels  = [], []
    for sample in batch_samples:
        batch_sentences.append(prompt(sample['comment']))
        batch_labels.append(int(sample['label']))
    batch_inputs = tokenizer(
        batch_sentences, 
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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

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
```

```
Using cuda device

Epoch 1/3
------------------------------
loss: 0.250585: 100%|█████████████████| 2400/2400 [03:02<00:00, 13.17it/s]
pos: 89.83 / 95.28 / 92.47, neg: 95.10 / 89.46 / 92.19
Macro-F1: 92.33 Micro-F1: 92.33

saving new weights...

Epoch 2/3
------------------------------
loss: 0.186270: 100%|█████████████████| 2400/2400 [02:52<00:00, 13.92it/s]
pos: 97.14 / 91.74 / 94.36, neg: 92.34 / 97.36 / 94.79
Macro-F1: 94.58 Micro-F1: 94.58

saving new weights...

Epoch 3/3
------------------------------
loss: 0.141027: 100%|█████████████████| 2400/2400 [02:52<00:00, 13.87it/s]
pos: 95.53 / 93.76 / 94.64, neg: 94.01 / 95.72 / 94.86
Macro-F1: 94.75 Micro-F1: 94.75

saving new weights...

Done!
```

可以看到，随着训练的进行，模型在验证集上的 Macro-F1 和 Micro-F1 值都在不断提升。因此 3 轮 Epoch 结束后，会在目录下保存 3 个模型权重：

```
epoch_1_valid_macrof1_92.331_microf1_92.329_model_weights.bin
epoch_2_valid_macrof1_94.575_microf1_94.577_model_weights.bin
epoch_3_valid_macrof1_94.748_microf1_94.749_model_weights.bin
```

至此，我们对 Prompt 情感分析模型的训练就完成了。

## 3. 测试模型

训练完成后，我们加载在验证集上性能最优的模型权重，汇报其在测试集上的性能，并且将模型的预测结果保存到文件中。

```python
import json

model.load_state_dict(torch.load('epoch_3_valid_macrof1_94.748_microf1_94.749_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, predictions, probs = [], [], []
    for batch_X, batch_mask_idx, label_word_id, batch_y in tqdm(test_dataloader):
        true_labels += batch_y.numpy().tolist()
        batch_X, batch_mask_idx = batch_X.to(device), batch_mask_idx.to(device)
        pred = model(batch_X, batch_mask_idx, label_word_id)
        predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
        probs += torch.nn.functional.softmax(pred, dim=-1)
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        save_resluts.append({
            "comment": test_data[s_idx]['comment'], 
            "label": true_labels[s_idx], 
            "pred": predictions[s_idx], 
            "prob": {'neg': probs[s_idx][0].item(), 'pos': probs[s_idx][1].item()}
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
```

```
evaluating on test set...
100%|█████████████████████████████████| 300/300 [00:06<00:00, 49.36it/s]
100%|█████████████████████████████████| 1200/1200 [00:00<00:00, 33764.90it/s]
pos: 96.79 / 94.24 / 95.50, neg: 94.24 / 96.79 / 95.50
Macro-F1: 95.50 Micro-F1: 95.50

saving predicted results...
```

可以看到，经过微调，模型在测试集上的 Macro-F1 值从 43.02 提升到 95.5，Micro-F1 值从 43.37 提升到 95.5，证明了我们对模型的微调是成功的。

我们打开保存预测结果的 *test_data_pred.json*，其中每一行对应一个样本，`comment` 对应评论，`label` 对应标注标签，`pred` 对应预测出的标签，`prediction` 对应具体预测出的概率值。

```
{
    "comment": "交通方便；环境很好；服务态度很好 房间较小", 
    "label": 1, 
    "pred": 1, 
    "prob": {"neg": 0.001537947915494442, "pos": 0.9984620809555054}
}
...
```

至此，我们使用 Transformers 库进行 Prompt 情感分析就全部完成了！

## 代码

与之前一样，我们按照功能将 Prompt 模型的代码拆分成模块并且存放在不同的文件中，整理后的代码存储在 Github：
[How-to-use-Transformers/src/text_cls_prompt_senti_chnsenticorp/](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/text_cls_prompt_senti_chnsenticorp)

运行 *run_prompt_senti_bert.sh* 脚本即可进行训练。如果要进行测试或者将模型预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的 Macro-F1 值和 Micro-F1 值都达到 95.25%（积极: 96.15 / 94.41 / 95.27, 消极: 94.36 / 96.11 / 95.23）（Nvidia Tesla V100, batch=4）。