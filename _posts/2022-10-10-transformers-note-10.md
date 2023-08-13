---
title: 第十三章：Prompting 情感分析
author: SHENG XU
date: 2022-10-10
category: NLP
layout: post
---

本文我们将运用 Transformers 库来完成情感分析任务，并且使用当前流行的 Prompting 方法。Prompting 方法的核心思想就是借助模板将问题转换为与预训练任务类似的形式来处理。

例如要判断标题“American Duo Wins Opening Beach Volleyball Match”的新闻类别，就可以应用模板“This is a $\texttt{[MASK]}$ News: $\{x\}$”将其转换为“This is a $\texttt{[MASK]}$ News: American Duo Wins Opening Beach Volleyball Match”，然后送入到包含 MLM (Mask Language Modeling) 预训练任务的模型中预测 $\texttt{[MASK]}$ 对应的词，最后将词映射到新闻类别（比如“Sports”对应“体育”类）。

> 如果你对 Prompting 概念不是很清楚，强烈建议先阅读一下[《Prompt 方法简介》](https://xiaosheng.blog/2022/09/10/what-is-prompt.html)。

下面我们以情感分析任务为例，运用 Transformers 库手工构建一个基于 Prompt 的模型来完成任务。

## 13.1 准备数据

这里我们选择中文情感分析语料库 ChnSentiCorp 作为数据集，其包含各类网络评论接近一万条，可以从百度 ERNIE [示例仓库](https://ernie-github.cdn.bcebos.com/data-chnsenticorp.tar.gz)或者[本仓库](https://github.com/jsksxs360/How-to-use-Transformers/blob/main/datasets/data-chnsenticorp.tar.gz)下载。

语料已经划分好了训练集、验证集、测试集（分别包含 9600、1200、1200 条评论），一行是一个样本，使用 `TAB` 分隔评论和对应的标签，“0”表示消极，“1”表示积极。例如：

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
{
    'comment': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 
    'label': '1'
}
```

最常见的 Prompting 方法就是借助模板将问题转换为 MLM 任务来解决。这里我们定义模板形式为“总体上来说很 $\texttt{[MASK]}$。$\{x\}$”，其中 $x$ 表示评论文本，并且规定如果 $\texttt{[MASK]}$ 被预测为“好”就判定情感为“积极”，如果预测为“差”就判定为“消极”，即“积极”和“消极”标签对应的 label word 分别为“好”和“差”。

可以看到，MLM 任务与[序列标注](/2022/03/18/transformers-note-6.html)任务很相似，也是对 token 进行分类，并且类别是整个词表，不同之处在于 MLM 任务只需要对文中特殊的 $\texttt{[MASK]}$ token 进行标注。因此在处理数据时我们需要：

1. 记录下模板中所有 $\texttt{[MASK]}$ token 位置，以便在模型的输出序列中将它们的表示取出。
2. 记录下 label word 对应的 token ID，因为我们实际上只关心模型在这些词语上的预测结果。

下面我们首先编写模板和 verbalizer 对应的函数：

```python
def get_prompt(x):
    prompt = f'总体上来说很[MASK]。{x}'
    return {
        'prompt': prompt, 
        'mask_offset': prompt.find('[MASK]')
    }

def get_verbalizer(tokenizer):
    return {
        'pos': {'token': '好', 'id': tokenizer.convert_tokens_to_ids("好")}, 
        'neg': {'token': '差', 'id': tokenizer.convert_tokens_to_ids("差")}
    }
```

这里由于模板中只包含一个 $\texttt{[MASK]}$ token，因此我们直接通过 `str.find()` 函数获取其位置，如果模板中包含多个 $\texttt{[MASK]}$ token，就需要把他们的位置都记录下来。verbalizer 记录了从标签到对应 label word 的映射，这里我们通过 `tokenizer.convert_tokens_to_ids()` 来获取 label word 对应的 token ID。例如，第一个样本转换后的模板为：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

comment = '这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。'

print('verbalizer:', get_verbalizer(tokenizer))

prompt_data = get_prompt(comment)
prompt, mask_offset = prompt_data['prompt'], prompt_data['mask_offset']

encoding = tokenizer(prompt, truncation=True)
tokens = encoding.tokens()
mask_idx = encoding.char_to_token(mask_offset)

print('prompt:', prompt)
print('prompt tokens:', tokens)
print('mask idx:', mask_idx)
```

```
verbalizer: {'pos': {'token': '好', 'id': 1962}, 'neg': {'token': '差', 'id': 2345}}
prompt: 总体上来说很[MASK]。这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般。
prompt tokens: ['[CLS]', '总', '体', '上', '来', '说', '很', '[MASK]', '。', '这', '个', '宾', '馆', '比', '较', '陈', '旧', '了', '，', '特', '价', '的', '房', '间', '也', '很', '一', '般', '。', '总', '体', '来', '说', '一', '般', '。', '[SEP]']
mask idx: 7
```

可以看到，BERT 分词器正确地将“[MASK]”识别为一个 token，并且记录下 $\texttt{[MASK]}$ token 在序列中的索引。

但是这种做法要求我们能够从词表中找到合适的 label word 来代表每一个类别，并且 label word 只能包含一个 token，而很多时候这是无法实现的。因此，另一种常见做法是为每个类别构建一个可学习的虚拟 token（又称伪 token），然后运用类别描述来初始化虚拟 token 的表示，最后使用这些虚拟 token 来扩展模型的 MLM 头。

例如，这里我们可以为“积极”和“消极”构建专门的虚拟 token “[POS]”和“[NEG]”，并且设置对应的类别描述为“好的、优秀的、正面的评价、积极的态度”和“差的、糟糕的、负面的评价、消极的态度”。下面我们扩展一下上面的 verbalizer 函数，添加一个 `vtype` 参数来区分两种 verbalizer 类型：

```python
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

vtype = 'virtual'
# add label words
if vtype == 'virtual':
    tokenizer.add_special_tokens({'additional_special_tokens': ['[POS]', '[NEG]']})
print('verbalizer:', get_verbalizer(tokenizer, vtype=vtype))
```

```
verbalizer: {
    'pos': {'token': '[POS]', 'id': 21128, 'description': '好的、优秀的、正面的评价、积极的态度'}, 
    'neg': {'token': '[NEG]', 'id': 21129, 'description': '差的、糟糕的、负面的评价、消极的态度'}
}
```

**注意：**“[POS]”和“[NEG]”是我们新添加的 token，因此如[模型与分词器](/intro/2021-12-11-transformers-note-2/)中介绍的那样，我们首先需要通过 `tokenizer.add_special_tokens()` 将这两个 token 添加进模型的词表，然后才能获取它们的 token ID。

Prompting 方法实际输入的是转换后的模板，而不是原始文本，因此我们首先使用模板函数 `get_prompt()` 来更新数据集：

```python
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
```

> 在实际应用场景下，模板的转换过程可能比文本中的要复杂得多（可能非常耗时），因此这里我们将其放置于数据集函数而不是 DataLoader 中，使得数据集返回的就是转换后的样本。

同样地，我们通过 `print(next(iter(train_data)))` 打印出一个训练样本：

```
{
    'comment': '选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 
    'prompt': '总体上来说很[MASK]。选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。酒店装修一般，但还算整洁。 泳池在大堂的屋顶，因此很小，不过女儿倒是喜欢。 包的早餐是西式的，还算丰富。 服务吗，一般', 
    'mask_offset': 6, 
    'label': '1'
}
```

可以看到输出的是转换后的模板，并且标记出了 $\texttt{[MASK]}$ 在文本中的位置，符合我们的预期。

### 数据预处理

与之前一样，接下来我们就通过 `DataLoader` 库来按批(batch)加载数据，将文本转换为模型可以接受的 token IDs。

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

vtype = 'base'
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if vtype == 'virtual':
    tokenizer.add_special_tokens({'additional_special_tokens': ['[POS]', '[NEG]']})

verbalizer = get_verbalizer(tokenizer, vtype='base')
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
        max_length=max_length, 
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

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

batch_data = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_data['batch_inputs'].items()})
print(batch_data['batch_inputs'])
print(batch_data['batch_mask_idxs'])
print(batch_data['label_word_id'])
print(batch_data['labels'])
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 201]), 
    'token_type_ids': torch.Size([4, 201]), 
    'attention_mask': torch.Size([4, 201])
}
{
    'input_ids': tensor([
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 6862, 2428, 2923,
         ...],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 3193, 2218, 2682,
         ...],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 2523, 4788, 4638,
         ...],
        [ 101, 2600,  860,  677, 3341, 6432, 2523,  103,  511, 3119, 1168, 6573,
         ...]
    ]), 
    'token_type_ids': tensor([...]), 
    'attention_mask': tensor([...])
}
[7, 7, 7, 7]
[2345, 1962]
[0, 1, 1, 1]
```

可以看到，DataLoader 按照我们设置的 batch size 每次对 4 个样本进行编码，将 token 序列填充到了相同的长度。这里由于我们对所有样本都添加相同的“前缀”，因此 `[MASK]` token 的索引都为 7。

> 这里我们设置 verbalizer 为普通类型，因此 `label_word_id` 为 `[2345, 1962]`，分别是“差”和“好”对应的 token ID。你也可以通过设置 `vtype = 'virtual'` 获取虚拟 token 类型的 verbalizer，此时模板不会有变化，但是 `label_word_id` 会变为 `[21129, 21128]`，分别对应我们添加的“[NEG]”和“[POS]” token。

## 13.2 训练模型

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
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def forward(self, batch_inputs, batch_mask_idxs, label_word_id, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idxs.unsqueeze(-1)).squeeze(1)
        prediction_scores = self.cls(batch_mask_reps)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(prediction_scores, labels)
        return loss, prediction_scores[:, label_word_id]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
config = AutoConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint, config=config).to(device)
if vtype == 'virtual':
    model.resize_token_embeddings(len(tokenizer))
    print(f"initialize embeddings of {verbalizer['pos']['token']} and {verbalizer['neg']['token']}")
    with torch.no_grad():
        pos_tokenized = tokenizer(verbalizer['pos']['description'])
        pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
        neg_tokenized = tokenizer(verbalizer['neg']['description'])
        neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
        new_embedding = model.bert.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
        new_embedding = model.bert.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)

print(model)
```

```
Using cpu device
initialize embeddings of [POS] and [NEG]
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

这里为了能够加载预训练好的 MLM head 参数，我们按照 Transformers 库中的模型结构来构建 `BertForPrompt` 模型。可以看到，BERT 自带的 MLM head 由两个部分组成：首先对所有 token 进行一个 $768 \times 768$ 的非线性映射（包括激活函数和 LayerNorm），然后使用一个 $768\times 21128$ 的线性映射预测词表中每个 token 的分数。

如果采用虚拟 label word，我们除了向模型词表中添加“[POS]”和“[NEG]” token 以外， 还按照我们在 verbalizer 中设置的描述来初始化这两个 token 的嵌入。这里我们首先运用分词器将描述文本转换为对应的 token 列表 $t_1,t_2,...,t_n$，然后初始化对应的表示为这些 token 嵌入的平均 $\frac{1}{n}\sum_{i=1}^n \boldsymbol{E}(t_i)$，$\boldsymbol{E}$ 就是模型的 token embedding 矩阵。

> 由于 BERT 原始词表包含 21128 个 token，因此最终 MLM head 进行的是一个 21128 类的分类任务。如果我们设置 `vtype = 'virtual'` 采用虚拟 token 版本的 verbalizer，就会向词表中添加“[POS]”和“[NEG]”使得词表大小增加到 21130，此时再次运行上面的代码，就会看到输出张量大小 `out_features` 变为 21130 了。

> **注意**
>
> 向模型词表中添加 token 包含两个步骤：
>
> 1. 通过 `tokenizer.add_special_tokens()` 向分词器中添加 token。这样分词器就能够在分词时将这些词分为独立的 token；
> 2. 通过 `model.resize_token_embeddings()` 扩展模型的词表大小。
>
> **这两个步骤缺一不可！**否则运行时就会出现错误。这里我们判断如果采用虚拟 label word，就调整模型词表大小。

> **注意**
>
> 与之前相比，本次我们构建的 BertForPrompt 模型中增加了两个特殊的函数：`get_output_embeddings()` 和 `set_output_embeddings()`，负责调整模型的 MLM head。
>
> 如果删除这两个函数，那么在调用 `model.resize_token_embeddings()` 时，就仅仅会调整模型词表的大小，而不会调整 MLM head，即运行上面的代码输出的张量维度依然是 21128。如果你不需要预测新添加 token 在 mask 位置的概率，那么即使删除这两个函数，代码也能正常运行，但是对于本文这种需要预测的情况就不行了。
>
> 因此，在绝大部分情况下，你都应该添加这两个函数。

为了让模型适配我们的任务，这里首先通过 `batched_index_select` 函数从 BERT 的输出序列中抽取出 $\texttt{[MASK]}$ token 对应的表示，在运用 MLM head 预测出该 $\texttt{[MASK]}$ token 对应词表中每个 token 的分数之后，我们只返回类别对应 label words 的分数用于分类。

为了测试模型的操作是否符合预期，我们尝试将一个 batch 的数据送入模型：

```python
def to_device(batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        elif k == 'label_word_id':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data

batch_data = next(iter(train_dataloader))
batch_data = to_device(batch_data)
_, outputs = model(**batch_data)
print(outputs.shape)
```

```
torch.Size([4, 2])
```

模型对每个样本都应该输出“消极”和“积极”两个类别对应 label word 的预测 logits 值，因此这里模型的输出尺寸 $4×2$  符合预期。

### 优化模型参数

与之前一样，我们将每一轮 Epoch 分为“训练循环”和“验证/测试循环”，在训练循环中计算损失、优化模型参数，在验证/测试循环中评估模型性能。下面我们首先实现训练循环。

因为对标签词的预测实际上就是对类别的预测，所以这里模型的输出与[同义句判断任务](https://transformers.run/intro/2021-12-17-transformers-note-4/)中介绍过的普通文本分类模型完全一致，损失也同样是通过在类别预测和答案标签之间计算交叉熵：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
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
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(batch_data)
            outputs = model(**batch_data)
            pred = outputs[1]
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

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_f1_score = 0.
for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}\n" + 30 * "-")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
    valid_scores = test_loop(valid_dataloader, model)
    macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
    f1_score = (macro_f1 + micro_f1) / 2
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{epoch+1}_valid_macrof1_{(macro_f1*100):0.3f}_microf1_{(micro_f1*100):0.3f}_model_weights.bin'
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
100%|█████████████████████████████████| 300/300 [01:46<00:00,  2.82it/s]
pos: 53.05 / 100.00 / 69.33, neg: 100.00 / 9.12 / 16.72
Macro-F1: 43.02 Micro-F1: 43.37
```

可以看到，得益于 Prompt 方法，不经微调的 BERT 模型也已经具有初步的情感分析能力，在测试集上的 Macro-F1 和 Micro-F1 值分别为 43.02 和 43.37。有趣的是，“积极”类别的召回率和“消极”类别的准确率都为 100%，这说明 BERT 对大部分样本都倾向于判断为“积极”类（可能预训练时看到的积极性文本更多吧）。

> **注意**
>
> 如果采用虚拟 label word（设置 `vtype = 'virtual'` ），模型是无法直接进行预测的。在扩展了词表之后，MLM head 的参数矩阵尺寸也会进行调整，新加入的参数都是随机初始化的，此时必须进行微调才能让 MLM head 正常工作。

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

vtype = 'base'
max_length = 512
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
                prompt_data = get_prompt(items[0])
                Data[idx] = {
                    'prompt': prompt_data['prompt'], 
                    'mask_offset': prompt_data['mask_offset'], 
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
if vtype == 'virtual':
    tokenizer.add_special_tokens({'additional_special_tokens': ['[POS]', '[NEG]']})

verbalizer = get_verbalizer(tokenizer, vtype=vtype)
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
        max_length=max_length, 
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
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def forward(self, batch_inputs, batch_mask_idxs, label_word_id, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        batch_mask_reps = batched_index_select(sequence_output, 1, batch_mask_idxs.unsqueeze(-1)).squeeze(1)
        pred_scores = self.cls(batch_mask_reps)[:, label_word_id]

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(pred_scores, labels)
        return loss, pred_scores

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPrompt.from_pretrained(checkpoint, config=config).to(device)
if vtype == 'virtual':
    model.resize_token_embeddings(len(tokenizer))
    print(f"initialize embeddings of {verbalizer['pos']['token']} and {verbalizer['neg']['token']}")
    with torch.no_grad():
        pos_tokenized = tokenizer(verbalizer['pos']['description'])
        pos_tokenized_ids = tokenizer.convert_tokens_to_ids(pos_tokenized)
        neg_tokenized = tokenizer(verbalizer['neg']['description'])
        neg_tokenized_ids = tokenizer.convert_tokens_to_ids(neg_tokenized)
        new_embedding = model.bert.embeddings.word_embeddings.weight[pos_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[pos_id, :] = new_embedding.clone().detach().requires_grad_(True)
        new_embedding = model.bert.embeddings.word_embeddings.weight[neg_tokenized_ids].mean(axis=0)
        model.bert.embeddings.word_embeddings.weight[neg_id, :] = new_embedding.clone().detach().requires_grad_(True)
  
def to_device(batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(device) for k_, v_ in v.items()
            }
        elif k == 'label_word_id':
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(device)
    return new_batch_data

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = epoch * len(dataloader)
    
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model):
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(batch_data)
            outputs = model(**batch_data)
            pred = outputs[1]
            predictions += pred.argmax(dim=-1).cpu().numpy().tolist()
    metrics = classification_report(true_labels, predictions, output_dict=True)
    pos_p, pos_r, pos_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
    neg_p, neg_r, neg_f1 = metrics['0']['precision'], metrics['0']['recall'], metrics['0']['f1-score']
    macro_f1, micro_f1 = metrics['macro avg']['f1-score'], metrics['weighted avg']['f1-score']
    print(f"pos: {pos_p*100:>0.2f} / {pos_r*100:>0.2f} / {pos_f1*100:>0.2f}, neg: {neg_p*100:>0.2f} / {neg_r*100:>0.2f} / {neg_f1*100:>0.2f}")
    print(f"Macro-F1: {macro_f1*100:>0.2f} Micro-F1: {micro_f1*100:>0.2f}\n")
    return metrics

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_f1_score = 0.
for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}\n" + 30 * "-")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
    valid_scores = test_loop(valid_dataloader, model)
    macro_f1, micro_f1 = valid_scores['macro avg']['f1-score'], valid_scores['weighted avg']['f1-score']
    f1_score = (macro_f1 + micro_f1) / 2
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        print('saving new weights...\n')
        torch.save(
            model.state_dict(), 
            f'epoch_{epoch+1}_valid_macrof1_{(macro_f1*100):0.3f}_microf1_{(micro_f1*100):0.3f}_model_weights.bin'
        )
print("Done!")
```

```
Using cuda device

Epoch 1/3
------------------------------
loss: 0.249279: 100%|█████████████████| 2400/2400 [01:42<00:00, 23.43it/s]
100%|█████████████████████████████████| 300/300 [00:03<00:00, 89.46it/s]
pos: 89.83 / 95.28 / 92.47, neg: 95.10 / 89.46 / 92.19
Macro-F1: 92.33 Micro-F1: 92.33

saving new weights...

Epoch 2/3
------------------------------
loss: 0.183906: 100%|█████████████████| 2400/2400 [01:42<00:00, 23.33it/s]
100%|█████████████████████████████████| 300/300 [00:03<00:00, 89.00it/s]
pos: 94.94 / 94.94 / 94.94, neg: 95.06 / 95.06 / 95.06
Macro-F1: 95.00 Micro-F1: 95.00

saving new weights...

Epoch 3/3
------------------------------
loss: 0.138809: 100%|█████████████████| 2400/2400 [01:42<00:00, 23.38it/s]
100%|█████████████████████████████████| 300/300 [00:03<00:00, 89.19it/s]
pos: 95.37 / 93.76 / 94.56, neg: 94.00 / 95.55 / 94.77
Macro-F1: 94.66 Micro-F1: 94.67

Done!
```

可以看到，随着训练的进行，模型在验证集上的 Macro-F1 和 Micro-F1 值先升后降。因此 3 轮 Epoch 结束后，会在目录下保存 2 个模型权重：

```
epoch_1_valid_macrof1_92.331_microf1_92.329_model_weights.bin
epoch_2_valid_macrof1_94.999_microf1_95.000_model_weights.bin
```

至此，我们对 Prompt 情感分析模型的训练就完成了。

## 13.3 测试模型

训练完成后，我们加载在验证集上性能最优的模型权重，汇报其在测试集上的性能。

```python
import json

model.load_state_dict(torch.load('epoch_2_valid_macrof1_94.999_microf1_95.000_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    true_labels, predictions, probs = [], [], []
    for batch_data in tqdm(test_dataloader):
        true_labels += batch_data['labels']
        batch_data = to_device(batch_data)
        outputs = model(**batch_data)
        pred = outputs[1]
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
100%|█████████████████████████████████| 300/300 [00:03<00:00, 89.36it/s]
100%|█████████████████████████████████| 1200/1200 [00:00<00:00, 68281.48it/s]
pos: 96.49 / 95.07 / 95.77, neg: 95.01 / 96.45 / 95.73
Macro-F1: 95.75 Micro-F1: 95.75

saving predicted results...
```

可以看到，经过微调，模型在测试集上的 Macro-F1 值从 43.02 提升到 95.75，Micro-F1 值从 43.37 提升到 95.75，证明了我们对模型的微调是成功的。

我们打开保存预测结果的 *test_data_pred.json*，其中每一行对应一个样本，`comment` 对应评论，`label` 对应标注标签，`pred` 对应预测出的标签，`prediction` 对应具体预测出的概率值。

```
{
    "comment": "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般", 
    "label": 1, 
    "pred": 0, 
    "prob": {
        "neg": 0.9691997766494751, 
        "pos": 0.030800189822912216
    }
}
...
```

至此，我们使用 Transformers 库进行 Prompt 情感分析就全部完成了！

> 当然，我们也可以通过设置 `vtype = 'virtual'` 使用虚拟 label words 来训练模型，最终在测试集上的性能为：
>
> ```
> pos: 96.14 / 94.24 / 95.18, neg: 94.21 / 96.11 / 95.15
> Macro-F1: 95.17 Micro-F1: 95.17
> ```
>
> 可以看到，在该任务上，使用虚拟 label words 取得了与基础 verbalizer 类似的性能。

## 13.4 封装预测函数

我们训练模型的目的是为了能够给其他人提供服务。尤其对于不熟悉深度学习的普通开发者而言，需要的只是一个能够完成特定任务的接口。因此在大多数情况下，我们都应该将模型的预测过程封装为一个端到端 (End-to-End) 的函数：输入文本，输出结果：

```python
def predict(model, tokenizer, comment, verbalizer):
    prompt_data = get_prompt(comment)
    prompt = prompt_data['prompt']
    encoding = tokenizer(prompt, truncation=True)
    mask_idx = encoding.char_to_token(prompt_data['mask_offset'])
    assert mask_idx is not None
    inputs = tokenizer(
        prompt, 
        max_length=max_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs, 
        'batch_mask_idxs': [mask_idx], 
        'label_word_id': [verbalizer['neg']['id'], verbalizer['pos']['id']] 
    }
    inputs = to_device(inputs)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
        prob = torch.nn.functional.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)[0].item()
    prob = prob[0][pred].item()
    return pred, prob
```

下面我们尝试输出模型对测试集前 5 条数据的预测结果：

```python
model.load_state_dict(torch.load('epoch_2_valid_macrof1_94.999_microf1_95.000_model_weights.bin'))

for i in range(5):
    data = test_data[i]
    pred, prob = predict(model, tokenizer, data['comment'], verbalizer)
    print(f"{data['comment']}\nlabel: {data['label']}\tpred: {pred}\tprob: {prob}")
```

```
这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
label: 1        pred: 0 prob: 0.9692065715789795
怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片！开始还怀疑是不是赠送的个别现象，可是后来发现每张DVD后面都有！真不知道生产商怎么想的，我想看的是猫和老鼠，不是米老鼠！如果厂家是想赠送的话，那就全套米老鼠和唐老鸭都赠送，只在每张DVD后面添加一集算什么？？简直是画蛇添足！！
label: 0        pred: 0 prob: 0.9736887216567993
还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。
label: 0        pred: 0 prob: 0.9987130165100098
交通方便；环境很好；服务态度很好 房间较小
label: 1        pred: 1 prob: 0.9942231774330139
不错，作者的观点很颠覆目前中国父母的教育方式，其实古人们对于教育已经有了很系统的体系了，可是现在的父母以及祖父母们更多的娇惯纵容孩子，放眼看去自私的孩子是大多数，父母觉得自己的孩子在外面只要不吃亏就是好事，完全把古人几千年总结的教育古训抛在的九霄云外。所以推荐准妈妈们可以在等待宝宝降临的时候，好好学习一下，怎么把孩子教育成一个有爱心、有责任心、宽容、大度的人。
label: 1        pred: 1 prob: 0.9959742426872253
```

可以看到，模型成功地输出了预测的类别和概率。

## 13.5 代码

本文的代码已整理至 [train_model_prompt_senti.py](https://github.com/jsksxs360/How-to-use-Transformers/blob/main/train_model_prompt_senti.py)

与之前一样，我们也按照标准项目结构将 Prompt 模型的代码按功能拆分成模块，整理后的代码存储在：
[How-to-use-Transformers/src/text_cls_prompt_senti_chnsenticorp/](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/text_cls_prompt_senti_chnsenticorp)

运行 *run_prompt_senti_bert.sh* 脚本即可进行训练。

```bash
bash run_prompt_senti_bert.sh
```

如果要进行测试或者将模型预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的结果为：
>
> ```
> ==> Nvidia GeForce RTX 3090, batch=4, vtype=base
> POS: 96.48 / 94.74 / 95.60, NEG: 94.69 / 96.45 / 95.56
> micro_F1 - 95.5835 macro_f1 - 95.5833
> 
> ==> Nvidia GeForce RTX 3090, batch=4, vtype=virtual
> POS: 96.79 / 94.08 / 95.41, NEG: 94.09 / 96.79 / 95.42
> micro_F1 - 95.4166 macro_f1 - 95.4167
> ```

