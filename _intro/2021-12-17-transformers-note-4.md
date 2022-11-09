---
title: 第七章：微调预训练模型
author: SHENG XU
date: 2021-12-17
category: NLP
layout: post
---

在上一篇[《必要的 Pytorch 知识》](/2021/12/14/transformers-note-3.html)中，我们介绍了使用 Transformers 库必须要掌握的 Pytorch 知识。 本文我们将正式上手微调一个句子对分类模型，并且保存验证集上最好的模型权重。

## 1. 加载数据集

我们以同义句判断任务为例（每次输入两个句子，判断它们是否为同义句），带大家构建我们的第一个 Transformers 模型。我们选择蚂蚁金融语义相似度数据集 [AFQMC](https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip) 作为语料，它提供了官方的数据划分，训练集 / 验证集 / 测试集分别包含 34334 / 4316 / 3861 个句子对，标签 0 表示非同义句，1 表示同义句：

```
{"sentence1": "还款还清了，为什么花呗账单显示还要还款", "sentence2": "花呗全额还清怎么显示没有还款", "label": "1"}
```

### Dataset

Pytorch 通过 `Dataset` 类和 `DataLoader` 类处理数据集和加载样本。同样地，这里我们首先继承 `Dataset` 类构造自定义数据集，以组织样本和标签。AFQMC 样本以 json 格式存储，因此我们使用 `json` 库按行读取样本，并且以行号作为索引构建数据集。

```python
from torch.utils.data import Dataset
import json

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('data/afqmc_public/train.json')
valid_data = AFQMC('data/afqmc_public/dev.json')

print(train_data[0])
```

```
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
```

可以看到，我们编写的 `AFQMC` 类成功读取了数据集，每一个样本都以字典形式保存，分别以 `sentence1`、`sentence2` 和 `label` 为键存储句子对和标签。

如果数据集非常巨大，难以一次性加载到内存中，我们也可以继承 `IterableDataset` 类构建迭代型数据集：

```python
from torch.utils.data import IterableDataset
import json

class IterableAFQMC(IterableDataset):
    def __init__(self, data_file):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, 'rt') as f:
            for line in f:
                sample = json.loads(line.strip())
                yield sample

train_data = IterableAFQMC('data/afqmc_public/train.json')

print(next(iter(train_data)))
```

```
{'sentence1': '蚂蚁借呗等额还款可以换成先息后本吗', 'sentence2': '借呗有先息到期还本吗', 'label': '0'}
```

### DataLoader

接下来就需要通过 `DataLoader` 库按批 (batch) 加载数据，并且将样本转换成模型可以接受的输入格式。对于 NLP 任务，这个环节就是将每个 batch 中的文本按照预训练模型的格式进行编码（包括 Padding、截断等操作）。

我们通过手工编写 DataLoader 的批处理函数 `collate_fn` 来实现。首先加载分词器，然后对每个 batch 中的所有句子对进行编码，同时把标签转换为张量格式：

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)

batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
```

```
batch_X shape: {
    'input_ids': torch.Size([4, 39]), 
    'token_type_ids': torch.Size([4, 39]), 
    'attention_mask': torch.Size([4, 39])
}
batch_y shape: torch.Size([4])

{'input_ids': tensor([
        [ 101, 5709, 1446, 5543, 3118,  802,  736, 3952, 3952, 2767, 1408,  102,
         3952, 2767, 1041,  966, 5543, 4500, 5709, 1446, 1408,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0],
        [ 101,  872, 8024, 2769, 6821, 5709, 1446, 4638, 7178, 6820, 2130,  749,
         8024, 6929, 2582,  720, 1357, 3867,  749,  102, 1963, 3362, 1357, 3867,
          749, 5709, 1446,  722, 1400, 8024, 1355, 4495, 4638, 6842, 3621, 2582,
          720, 1215,  102],
        [ 101, 1963,  862, 2990, 7770,  955, 1446,  677, 7361,  102, 6010, 6009,
          955, 1446, 1963,  862,  712, 1220, 2990, 7583, 2428,  102,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0],
        [ 101, 2582, 3416, 2990, 7770,  955, 1446, 7583, 2428,  102,  955, 1446,
         2990, 4157, 7583, 2428, 1416,  102,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0]]), 
 'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
tensor([1, 0, 1, 1])
```

可以看到，DataLoader 按照我们设置的 batch size 每次对 4 个样本进行编码，并且通过设置 `padding=True` 和 `truncation=True` 来自动对每个 batch 中的样本进行补全和截断。这里我们选择 BERT 模型作为 checkpoint，所以每个样本都被处理成了“$\texttt{[CLS]}\text{ sen1 }\texttt{[SEP]}\text{ sen2 }\texttt{[SEP]}$”的形式。

> 这种只在一个 batch 内进行补全的操作被称为动态补全 (Dynamic padding)，Hugging Face 也提供了 `DataCollatorWithPadding` 类来进行，如果感兴趣可以自行[了解](https://huggingface.co/course/chapter3/2?fw=pt#dynamic-padding)。

## 2. 训练模型

### 构建模型

对于分类任务，可以直接使用我们前面介绍过的 `AutoModelForSequenceClassification` 类来完成。但是在实际操作中，除了使用预训练模型编码文本外，我们通常还会进行许多自定义操作，因此在大部分情况下我们都需要自己编写模型。

最简单的方式是首先利用 Transformers 库加载 BERT 模型，然后接一个全连接层完成分类：

```python
from torch import nn
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super(BertForPairwiseCLS, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

model = BertForPairwiseCLS().to(device)
print(model)
```

```python
Using cpu device
NeuralNetwork(
  (bert_encoder): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(...)
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

这里模型首先将输入送入到 BERT 模型中，将每一个 token 都编码为维度为 768 的向量，然后从输出序列中取出第一个 `[CLS]` token 的编码表示作为整个句子对的语义表示，再送入到一个线性全连接层中预测两个类别的分数。这种方式简单粗暴，但是相当于在 Transformers 模型外又包了一层，因此无法再调用 Transformers 库预置的模型函数。

更为常见的写法是继承 Transformers 库中的预训练模型来创建自己的模型。例如这里我们可以继承 BERT 模型（`BertPreTrainedModel` 类）来创建一个与上面模型结构完全相同的分类器：

```python
from torch import nn
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        cls_vectors = bert_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)
print(model)
```

注意，此时我们的模型是 Transformers 预训练模型的子类，因此需要通过预置的 `from_pretrained` 函数来加载模型参数。这种方式也使得我们可以更灵活地操作模型细节，例如这里 Dropout 层就可以直接加载 BERT 模型自带的参数值，而不用像上面一样手工赋值。

为了确保模型的输出符合我们的预期，我们尝试将一个 Batch 的数据送入模型：

```python
outputs = model(batch_X)
print(outputs.shape)
```

```
torch.Size([4, 2])
```

可以看到模型输出了一个 $4 \times 2$ 的张量，符合我们的预期（每个样本输出 2 维的 logits 值分别表示两个类别的预测分数，batch 内共 4 个样本）。

### 优化模型参数

正如之前介绍的那样，在训练模型时，我们将每一轮 Epoch 分为训练循环和验证/测试循环。在训练循环中计算损失、优化模型的参数，在验证/测试循环中评估模型的性能：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
```

最后，将”训练循环”和”验证/测试循环”组合成 Epoch，就可以进行模型的训练和验证了。

与 Pytorch 类似，Transformers 库同样实现了很多的优化器，并且相比 Pytorch 固定学习率，Transformers 库的优化器会随着训练过程逐步减小学习率（通常会产生更好的效果）。例如我们前面使用过的 AdamW 优化器：

```python
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
```

默认情况下，优化器会线性衰减学习率，对于上面的例子，学习率会线性地从 $\text{5e-5}$ 降到 $0$。为了正确地定义学习率调度器，我们需要知道总的训练步数 (step)，它等于训练轮数 (Epoch number) 乘以每一轮中的步数（也就是训练 dataloader 的大小）：

```python
from transformers import get_scheduler

epochs = 3
num_training_steps = epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```

```
25752
```

完整的训练过程如下（下面结果均为使用 Tesla V100 GPU 训练）：

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
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    test_loop(valid_dataloader, model, mode='Valid')
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.552296: 100%|█████████| 8584/8584 [07:16<00:00, 19.65it/s]
Valid Accuracy: 72.1%

Epoch 2/3
-------------------------------
loss: 0.501410: 100%|█████████| 8584/8584 [07:16<00:00, 19.66it/s]
Valid Accuracy: 73.0%

Epoch 3/3
-------------------------------
loss: 0.450708: 100%|█████████| 8584/8584 [07:15<00:00, 19.70it/s]
Valid Accuracy: 74.1%

Done!
```

### 保存和加载模型

在大多数情况下，我们还需要根据验证集上的表现来调整超参数以及选出最好的模型，最后再将选出的模型应用于测试集以评估性能。这里我们在测试循环时返回计算出的准确率，然后对上面的 Epoch 训练代码进行小幅的调整，以保存验证集上准确率最高的模型：

```python
def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
```

```
Using cuda device

Epoch 1/3
-------------------------------
loss: 0.556518: 100%|█████████| 8584/8584 [07:51<00:00, 18.20it/s]
Valid Accuracy: 71.8%

saving new weights...

Epoch 2/3
-------------------------------
loss: 0.506202: 100%|█████████| 8584/8584 [07:15<00:00, 19.71it/s]
Valid Accuracy: 72.0%

saving new weights...

Epoch 3/3
-------------------------------
loss: 0.455851: 100%|█████████| 8584/8584 [07:16<00:00, 19.68it/s]
Valid Accuracy: 74.1%

saving new weights...

Done!
```

可以看到，随着训练的进行，在验证集上的准确率逐步提升（71.8% -> 72.0% -> 74.1%）。因此，3 轮 Epoch 训练结束后，会在目录下保存下所有三轮模型的权重：

```
epoch_1_valid_acc_71.8_model_weights.bin
epoch_2_valid_acc_72.0_model_weights.bin
epoch_3_valid_acc_74.1_model_weights.bin
```

至此，我们手工构建的文本分类模型的训练过程就完成了，完整的训练代码如下：

```python
import random
import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
seed_everything(42)

learning_rate = 1e-5
batch_size = 4
epoch_num = 3

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class AFQMC(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = AFQMC('data/afqmc_public/train.json')
valid_data = AFQMC('data/afqmc_public/dev.json')

def collote_fn(batch_samples):
    batch_sentence_1, batch_sentence_2 = [], []
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_sentence_1, 
        batch_sentence_2, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn)
valid_dataloader= DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn)

class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()
    
    def forward(self, x):
        outputs = self.bert(**x)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
model = BertForPairwiseCLS.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)
    
    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_acc = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
```

这里我们还通过 `seed_everything` 函数手工设置训练过程中所有的随机数种子为 42，从而使得模型结果可以复现，而不是每次运行代码都是不同的结果。

最后，我们加载验证集上最优的模型权重，汇报其在测试集上的性能。由于 AFQMC 公布的测试集上并没有标签，无法评估性能，这里我们暂且用验证集代替进行演示：

```python
model.load_state_dict(torch.load('epoch_3_valid_acc_74.1_model_weights.bin'))
test_loop(valid_dataloader, model, mode='Test')
```

```
Test Accuracy: 74.1%
```

> 注意：前面我们只保存了模型的权重，因此如果要单独调用上面的代码，需要首先实例化一个结构完全一样的模型，再通过 `model.load_state_dict()` 函数加载权重。

最终在测试集（这里用了验证集）上的准确率为 74.1%，与前面汇报的一致，也验证了模型参数的加载过程是正确的。

## 代码

出于演示需要，本文将所有代码都写在同一个文件中，但是如果模型结构复杂或者逻辑操作很多，这种方式就会大大降低代码可读性。因此，我们从一开始就应该养成切分代码的好习惯，尤其对于深度学习项目而言，合理规划的项目结构能够使得其他研究人员更加容易地复现出汇报的结果。

本文整理后的代码存储在 Github，大家在构建自己的项目时也可以参考该框架：  
[How-to-use-Transformers/src/pairwise_cls_similarity_afqmc/](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/pairwise_cls_similarity_afqmc)

与 Transformers 库类似，我们将模型损失的计算也包含进模型本身，这样在训练循环中我们就可以直接使用模型返回的损失进行反向传播。

这里我们同时加载 BERT 和 RoBERTa 权重来构建分类器，分别通过运行 *run_simi_bert.sh* 和 *run_simi_roberta.sh* 脚本进行训练。如果要进行测试或者将预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 在测试集（验证集）上的准确率为 73.61%，RoBERTa 为 73.84%（Nvidia Tesla V100, batch=16）。

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://pytorch.org/tutorials/beginner/basics/intro.html) Pytorch 在线教程



