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

```
train set size: 9600
valid set size: 1200
test set size: 1200
{'comment': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 'label': '1'}
```

### 数据预处理

接下来我们就需要通过 `DataLoader` 库来按 batch 加载数据，将文本转换为模型可以接受的 token IDs。

大部分 Prompt 方法都是通过模板将问题转换为 MLM 任务的形式来解决，同样地，这里我们定义模板为“总体上来说很 $\texttt{[MASK]}$。$\text{x}$”，并且规定如果 $\texttt{[MASK]}$ 预测为 token “好”就判定情感为“积极”，如果预测为 token “差”就判定为“消极”。

MLM 任务与[序列标注](/2022/03/18/transformers-note-6.html)任务很相似，也是对 token 进行分类，并且类别是整个词表，不同之处在于 MLM 任务只对文中特殊的 $\texttt{[MASK]}$ token 进行标注。因此 MLM 任务的标签同样是一个序列，但是只有 $\texttt{[MASK]}$ token 的位置为对应词语的索引，其他位置都应该设为 -100，以便在使用交叉熵计算损失时忽略它们。

下面以处理第一个样本为例。我们通过 `char_to_token()` 将 $\texttt{[MASK]}$ 从原文位置映射到切分后的 token 索引，并且根据情感极性将对应的标签设为“好”或“差”的 token ID。

```python
from transformers import AutoTokenizer
import numpy as np

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
labels = np.full(len(tokens), -100)
mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
labels[mask_idx] = pos_id if label == '1' else neg_id

print(tokens)
print(labels)
```

```
['[CLS]', '总', '体', '上', '来', '说', '很', '[MASK]', '。', '这', '个', '宾', '馆', '比', '较', '陈', '旧', '了', '，', '特', '价', '的', '房', '间', '也', '很', '一', '般', '。', '总', '体', '来', '说', '一', '般', '。', '[SEP]']
[-100 -100 -100 -100 -100 -100 -100 1962 -100 -100 -100 -100 -100 -100
 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
 -100 -100 -100 -100 -100 -100 -100 -100 -100]
```

可以看到，BERT 分词器正确地将“[MASK]”识别为一个 token，并且将 `[MASK]` token 对应的标签设置为“好”的 token ID。

在实际编写 DataLoader 的批处理函数 `collate_fn()` 时，我们处理的不是一个而是多个样本，因此需要对上面的操作进行扩展。

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

pre_text = '总体上来说很[MASK]。'
pos_id = tokenizer.convert_tokens_to_ids("好")
neg_id = tokenizer.convert_tokens_to_ids("差")

def collote_fn(batch_samples):
    batch_sentence, batch_senti  = [], []
    for sample in batch_samples:
        batch_sentence.append(pre_text + sample['comment'])
        batch_senti.append(sample['label'])
    batch_inputs = tokenizer(
        batch_sentence, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    batch_label = np.full(batch_inputs['input_ids'].shape, -100)
    for s_idx, sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence, truncation=True)
        mask_idx = encoding.char_to_token(sentence.find('[MASK]'))
        batch_label[s_idx][mask_idx] = pos_id if batch_senti[s_idx] == 1 else neg_id
    return batch_inputs, torch.tensor(batch_label)

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 96]), 
    'token_type_ids': torch.Size([4, 96]), 
    'attention_mask': torch.Size([4, 96])
}
batch_y shape: torch.Size([4, 96])

{'input_ids': tensor([
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 2523,  671, 5663,
         8024, 6432, 2124, 7410, 1416, 8024, 3300,  763, 1296, 6404, 1348, 2523,
         5042, 1296, 8024,  784,  720,  100,  117,  100,  119, 6432, 2124, 5042,
         1296, 1416, 8024, 2523, 1914, 1296, 6404, 1348, 2523, 7410, 8024,  784,
          720, 3661,  811, 7667, 5401, 2159, 2360, 8024, 5455, 7965, 1590, 4906,
         1278, 4495,  722, 5102, 4638, 8024, 1353, 3633,  679, 2743, 4638, 4385,
         1762,  738, 3766, 6381,  857, 8024, 2743, 4638, 6820, 3221, 2743, 4638,
          511, 2600,  860, 6432, 3341, 8024, 3766,  784,  720, 2692, 2590,  102],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 2791, 7313, 2397,
         1112, 5653, 3302,  117, 4958, 3698, 3837, 1220,  738, 2523, 1962,  511,
         3766, 3300,  679, 5679, 3698, 1456,  119,  119,  119, 4507,  754, 1905,
          754, 7317, 2356, 1277,  117,  769, 6858, 3683, 6772, 3175,  912, 8024,
          852, 1398, 3198,  738, 3300, 4157, 1648, 3325,  119,  119,  119,  102,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 2791, 7313, 3440,
         3613, 1469, 4384, 1862, 5318, 2190,  126, 3215,  119, 7649, 5831, 6574,
         7030, 2247,  754,  677, 5023,  119, 1372, 3221, 4895, 2458, 2356,  704,
         2552, 6772, 6823,  119,  679, 6814, 6983, 2421, 3300, 4408, 6756,  119,
         1963, 5543, 3022,  677, 4408, 6756,  117, 1156, 1282, 1059, 1282, 5401,
          119,  102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 2791, 7313, 1377,
          809,  117, 4294, 1166, 4638, 1947, 2791,  117, 7478, 2382,  679, 7231,
          117, 2218, 3221, 7623, 1324, 1922, 5552,  749,  117, 3193, 7623, 3018,
         2533, 3766, 5517, 1366,  117,  102,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0]]), 
 'token_type_ids': tensor(...), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

tensor([[-100, -100, -100, -100, -100, -100, -100, 2345, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100, -100, -100, 1962, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100, -100, -100, 1962, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
        [-100, -100, -100, -100, -100, -100, -100, 1962, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
```

可以看到，DataLoader 按照我们设置的 batch size 每次对 4 个样本进行编码，将 token 序列填充到了相同的长度。标签中 `[MASK]` token 对应的索引都转换为了情感极性对应“好”或“差”的 token ID。

> 这里我们对所有样本都应用相同的模板，添加相同的“前缀”，因此 `[MASK]` token 的位置其实是固定的，我们不必对每个样本都单独计算 `[MASK]`对应的 token 位置。
>
> 在实际操作中，我们既可以对样本应用不同的模板，也可以将 `[MASK]` 插入到样本中的任意位置，甚至模板中可以包含多个 `[MASK]`，需要根据实际情况对数据预处理进行调整。

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

注意，这里为了能够加载预训练好的 MLM head 参数，我们严格按照 Transformers 库中的模型结构来构建 `BertForPrompt` 模型。可以看到，BERT 自带的 MLM head 由两个部分组成：首先对所有 token 进行一个 $768 \times 768$ 的非线性映射（包括激活函数和 LayerNorm），然后使用一个 $768\times 21128$ 的线性映射预测词表中每个 token 的分数。

为了测试模型的操作是否符合预期，我们尝试将一个 batch 的数据送入模型：

```python
outputs = model(batch_X)
print(outputs.shape)
```

```
torch.Size([4, 96, 21128])
```

对于 batch 内 4 个都被填充到长度为 $96$ 的样本，模型对每个 token 都应该输出一个词表大小的向量（对应词表中每个词语的预测 logits 值），因此这里模型的输出尺寸 $4\times 96\times 21128$ 完全符合预期。

### 训练循环

与之前一样，我们将每一轮 Epoch 分为“训练循环”和“验证/测试循环”，在训练循环中计算损失、优化模型参数，在验证/测试循环中评估模型性能。下面我们首先实现训练循环。

MLM 任务计算损失的方式与序列标注任务几乎完全一致，同样也是在标签序列和预测序列之间计算交叉熵损失，唯一的区别是 MLM 任务只需要计算 `[MASK]` token 位置的损失：

```python
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
```

### 后处理

因为我们最终需要的是情感标签，所以在编写“验证/测试循环”之前，我们先讨论一下 Prompt 模型的后处理——怎么将模型的输出转换为情感标签。

上面我们介绍过，在 MLM 模型的输出中，我们只关注 `[MASK]` token 的预测值，并且只关心其中特定几个表意词的概率值。例如对于情感分析任务，我们只关心预测出的“好”和“坏”两个词的 logit 值的谁更大，如果“好”大于“差”对应的情感标签就是积极，反之就是消极。

因为 Prompt 方法可以在不微调模型的情况下进行预测，这里我们使用 BERT 模型直接对验证集上的前 12 个样本进行预测以展示后处理过程：

```python
valid_data = ChnSentiCorp('data/ChnSentiCorp/dev.txt')
small_eval_set = [valid_data[idx] for idx in range(12)]

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
eval_set = DataLoader(small_eval_set, batch_size=4, shuffle=False, collate_fn=collote_fn)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
```

接下来，与之前任务中的验证/测试循环一样，在 `torch.no_grad()` 上下文管理器下，我们使用模型对所有样本进行预测，并且汇总预测出的“好”和“差” token 对应的 logits 值：

```python
pos_id = tokenizer.convert_tokens_to_ids("好")
neg_id = tokenizer.convert_tokens_to_ids("差")
results = []
model.eval()
for batch_data, _ in eval_set:
    batch_data = batch_data.to(device)
    with torch.no_grad():
        token_logits = model(**batch_data).logits
    mask_token_indexs = torch.where(batch_data["input_ids"] == tokenizer.mask_token_id)[1]
    for s_idx, mask_idx in enumerate(mask_token_indexs):
        results.append(token_logits[s_idx, mask_idx, [neg_id, pos_id]].cpu().numpy())
```

最后我们遍历数据集中的样本，使用 `softmax` 函数将 logits 值转换为概率值，并且同步打印预测和标注结果来进行对比：

```python
true_labels, true_predictions = [], []
for s_idx, example in enumerate(small_eval_set):
    comment = example['comment']
    label = example['label']
    probs = torch.nn.functional.softmax(torch.tensor(results[s_idx]), dim=-1)
    print(comment, label)
    print('pred:', {'0': probs[0].item(), '1': probs[1].item()})
    true_labels.append(int(label))
    true_predictions.append(0 if probs[0] > probs[1] else 1)
```

```
這間酒店環境和服務態度亦算不錯,但房間空間太小~~不宣容納太大件行李~~且房間格調還可以~~ 中餐廳的廣東點心不太好吃~~要改善之~~~~但算價錢平宜~~可接受~~ 西餐廳格調都很好~~但吃的味道一般且令人等得太耐了~~要改善之~~ 1
pred: {'0': 0.0033), '1': 0.9967}
<荐书> 推荐所有喜欢<红楼>的红迷们一定要收藏这本书,要知道当年我听说这本书的时候花很长时间去图书馆找和借都没能如愿,所以这次一看到当当有,马上买了,红迷们也要记得备货哦! 1
pred: {'0': 0.0003), '1': 0.9997}
...
```

对于分类任务最常见的就是通过精确率、召回率、F1值 (P / R / F1) 指标来评估每个类别的预测性能，然后再通过宏/微 F1 值 (Macro-F1/Micro-F1) 来评估整体分类性能。这里我们借助机器学习包 [sklearn](https://scikit-learn.org/stable/#) 提供的 `classification_report` 函数来输出这些指标：

```python
from sklearn.metrics import classification_report
print(classification_report(true_labels, true_predictions, output_dict=False))
```

```
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         3
           1       0.75      1.00      0.86         9

    accuracy                           0.75        12
   macro avg       0.38      0.50      0.43        12
weighted avg       0.56      0.75      0.64        12
```

可以看到，这里模型将 12 个样本都预测为了“积极”类（标签 1），因此该类别的召回率为 100%，而“消极”类的指标都为 0，代表整体性能的 Macro-F1/Micro-F1 值只有 0.43 和 0.64。

### 测试循环

熟悉了后处理操作之后，编写验证/测试循环就很简单了，只需对上面的这些步骤稍作整合即可：

```python
from sklearn.metrics import classification_report

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
```

为了方便后续保存验证集上最好的模型，这里我们还在验证/测试循环中返回评估结果。

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
    valid_scores = test_loop(valid_dataloader, valid_data, model)
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

test_loop(test_dataloader, test_data, model)
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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

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
    valid_scores = test_loop(valid_dataloader, valid_data, model)
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
loss: 0.258182: 100%|█████████████████| 2400/2400 [03:05<00:00, 12.96it/s]
100%|█████████████████████████████████| 300/300 [00:06<00:00, 44.98it/s]
pos: 90.81 / 94.94 / 92.83, neg: 94.83 / 90.61 / 92.67
Macro-F1: 92.75 Micro-F1: 92.75

saving new weights...

Epoch 2/3
------------------------------
loss: 0.190014: 100%|█████████████████| 2400/2400 [03:04<00:00, 12.98it/s]
100%|█████████████████████████████████| 300/300 [00:06<00:00, 44.79it/s]
pos: 94.88 / 93.76 / 94.32, neg: 93.97 / 95.06 / 94.51
Macro-F1: 94.41 Micro-F1: 94.42

saving new weights...

Epoch 3/3
------------------------------
loss: 0.143803: 100%|█████████████████| 2400/2400 [03:04<00:00, 13.03it/s]
100%|█████████████████████████████████| 300/300 [00:06<00:00, 44.29it/s]
pos: 96.05 / 94.44 / 95.24, neg: 94.65 / 96.21 / 95.42
Macro-F1: 95.33 Micro-F1: 95.33

saving new weights...

Done!
```

可以看到，随着训练的进行，模型在验证集上的 Macro-F1 和 Micro-F1 值都在不断提升。因此 3 轮 Epoch 结束后，会在目录下保存 3 个模型权重：

```
epoch_1_valid_macrof1_92.749_microf1_92.748_model_weights.bin
epoch_2_valid_macrof1_94.415_microf1_94.416_model_weights.bin
epoch_3_valid_macrof1_95.331_microf1_95.333_model_weights.bin
```

至此，我们对 Prompt 情感分析模型的训练就完成了。

## 3. 测试模型

训练完成后，我们加载在验证集上性能最优的模型权重，汇报其在测试集上的性能，并且将模型的预测结果保存到文件中。

```python
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
```

```
evaluating on test set...
100%|█████████████████████████████████| 300/300 [00:06<00:00, 44.09it/s]
100%|█████████████████████████████████| 1200/1200 [00:00<00:00, 47667.51it/s]
pos: 96.46 / 94.08 / 95.25, neg: 94.07 / 96.45 / 95.25
Macro-F1: 95.25 Micro-F1: 95.25

saving predicted results...
```

可以看到，经过微调，模型在测试集上的 Macro-F1 值从 43.02 提升到 95.25，Micro-F1 值从 43.37 提升到 95.25，证明了我们对模型的微调是成功的。

我们打开保存预测结果的 *test_data_pred.json*，其中每一行对应一个样本，`comment` 对应评论，`label` 对应标注标签，`pred` 对应预测出的标签，`prediction` 对应具体预测出的概率值。

```
{
    "comment": "交通方便；环境很好；服务态度很好 房间较小", 
    "label": "1", 
    "pred": "1", 
    "prediction": {"0": 0.002953010145574808, "1": 0.9970470070838928}
}
...
```

至此，我们使用 Transformers 库进行 Prompt 情感分析就全部完成了！

## 代码

与之前一样，我们按照功能将 Prompt 模型的代码拆分成模块并且存放在不同的文件中，整理后的代码存储在 Github：
[How-to-use-Transformers/src/text_cls_prompt_senti_chnsenticorp/](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/text_cls_prompt_senti_chnsenticorp)

运行 *run_prompt_senti_bert.sh* 脚本即可进行训练。如果要进行测试或者将模型预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的 Macro-F1 值和 Micro-F1 值都达到  95.33%（积极: 96.62 / 94.08 / 95.33, 消极: 94.08 / 96.62 / 95.33） （Nvidia Tesla V100, batch=4）。