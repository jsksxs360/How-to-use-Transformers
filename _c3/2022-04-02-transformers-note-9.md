---
title: 第十二章：抽取式问答
author: SHENG XU
date: 2022-04-02
category: NLP
layout: post
---

本文我们将运用 Transformers 库来完成抽取式问答任务。自动问答 (Question Answering, QA) 是经典的 NLP 任务，需要模型基于给定的上下文回答问题。

根据回答方式的不同可以分为：

- **抽取式 (extractive) 问答：**从上下文中截取片段作为回答，类似于我们前面介绍的[序列标注](/2022/03/18/transformers-note-6.html)任务；
- **生成式 (generative) 问答：**生成一个文本片段作为回答，类似于我们前面介绍的[翻译](/2022/03/24/transformers-note-7.html)和[摘要](/2022/03/29/transformers-note-8.html)任务。

抽取式问答模型通常采用纯 Encoder 框架（例如 BERT），它更适用于处理事实性问题，例如“谁发明了 Transformer 架构？”，这些问题的答案通常就包含在上下文中；而生成式问答模型则通常采用 Encoder-Decoder 框架（例如 T5、BART），它更适用于处理开放式问题，例如“天空为什么是蓝色的？”，这些问题的答案通常需要结合上下文语义再进行抽象表达。

本文我们将微调一个 BERT 模型来完成抽取式问答任务：对于给定的问题，从上下文中抽取出概率最大的文本片段作为答案。

> 如果你对生成式问答感兴趣，可以参考 Hugging Face 提供的基于 [ELI5](https://huggingface.co/datasets/eli5) 数据库的 [Demo](https://yjernite.github.io/lfqa.html)。

## 1. 准备数据

我们选择由哈工大讯飞联合实验室构建的中文阅读理解语料库 [CMRC 2018](https://ymcui.com/cmrc2018/) 作为数据集，该语料是一个类似于 [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) 的抽取式数据集，对于每个问题都从原文中截取片段 (span) 作为答案，可以从 [Github](https://github.com/ymcui/cmrc2018/tree/master/squad-style-data) 下载。

其中 *cmrc2018_train.json*、*cmrc2018_dev.json* 和 *cmrc2018_trial.json* 分别对应训练集、验证集和测试集。对于每篇文章，CMRC 2018 都标注了一些问题以及对应的答案（包括答案的文本和位置），例如：

```
{
 "context": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。...", 
 "qas": [{
     "question": "《战国无双3》是由哪两个公司合作开发的？", 
     "id": "DEV_0_QUERY_0", 
     "answers": [{
         "text": "光荣和ω-force", 
         "answer_start": 11
     }, {
         "text": "光荣和ω-force", 
         "answer_start": 11
     }, {
         "text": "光荣和ω-force", 
         "answer_start": 11
     }]
 }, {
     "question": "男女主角亦有专属声优这一模式是由谁改编的？", 
     "id": "DEV_0_QUERY_1", 
     "answers": [{
         "text": "村雨城", 
         "answer_start": 226
     }, {
         "text": "村雨城", 
         "answer_start": 226
     }, {
         "text": "任天堂游戏谜之村雨城", 
         "answer_start": 219
     }]
 }, ...
 ]
}
```

一个问题可能对应有多个参考答案，在训练时我们任意选择其中一个作为标签，在验证/测试时，我们则将预测答案和所有参考答案都送入打分函数来评估模型的性能。

### 构建数据集

与之前一样，我们首先编写继承自 `Dataset` 类的自定义数据集用于组织样本和标签。原始数据集中一个样本对应一个上下文，这里我们将它调整为一个问题一个样本，参考答案则处理为包含 `text` 和 `answer_start` 字段的字典，分别存储答案文本和位置：

```python
from torch.utils.data import Dataset
import json

class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = CMRC2018('data/cmrc2018/cmrc2018_train.json')
valid_data = CMRC2018('data/cmrc2018/cmrc2018_dev.json')
test_data = CMRC2018('data/cmrc2018/cmrc2018_trial.json')
```

下面我们输出数据集的尺寸，并且打印出一个验证样本：

```python
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(valid_data)))
```

```
train set size: 10142
valid set size: 3219
test set size: 1002

{
  'id': 'DEV_0_QUERY_0', 
  'title': '战国无双3', 
  'context': '《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品', 
  'question': '《战国无双3》是由哪两个公司合作开发的？', 
  'answers': {
    'text': ['光荣和ω-force', '光荣和ω-force', '光荣和ω-force'], 
    'answer_start': [11, 11, 11]
  }
}
```

可以数据集处理为了我们预期的格式，因为参考答案可以有多个，所以答案文本 `text` 和位置 `answer_start` 都是列表。

### 数据预处理

接下来，我们就需要通过 `DataLoader` 库按 batch 加载数据，将文本转换为模型可以接受的 token IDs，并且构建对应的标签，标记答案在上下文中起始和结束位置。本文使用 BERT 模型来完成任务，因此我们首先加载对应的分词器：

```python
from transformers import AutoTokenizer

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

正如我们之前在[快速分词器](/2022/03/08/transformers-note-5.html#3-抽取式问答任务)中介绍过的那样，对于抽取式问答任务，我们会将问题和上下文编码为下面的形式：

```
[CLS] question [SEP] context [SEP]
```

标签是答案在上下文中起始/结束 token 的索引，模型的任务就是预测每个 token 为答案片段的起始/结束的概率，即为每个 token 预测一个起始 logit 值和结束 logit 值。例如对于下面的文本，理想标签为：

![qa_labels](/assets/img/transformers-note-9/qa_labels.svg)

我们在[问答 pipeline](/2022/03/08/transformers-note-5.html#处理长文本) 中就讨论过，由于问题与上下文拼接后的 token 序列可能超过模型的最大输入长度，因此我们可以将上下文切分为短文本块 (chunk) 来处理，同时为了避免答案被截断，我们使用滑窗使得切分出的文本块之间有重叠。

**如果对分块操作感到陌生，可以参见快速分词器中的[处理长文本](/2022/03/08/transformers-note-5.html#处理长文本)小节，下面只做简单回顾。**

下面我们尝试编码第一个训练样本，将拼接后的最大序列长度设为 300，滑窗大小设为 50，只需要给分词器传递以下参数：

- `max_length`：设置编码后的最大序列长度（这里设为 300）；
- `truncation="only_second"`：只截断第二个输入，这里上下文是第二个输入；
- `stride`：两个相邻文本块之间的重合 token 数量（这里设为 50）；
- `return_overflowing_tokens=True`：允许分词器返回重叠 token。

```python
context = train_data[0]["context"]
question = train_data[0]["question"]

inputs = tokenizer(
    question,
    context,
    max_length=300,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
```

```
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 范 廷 颂 枢 机 （ ， ） ， 圣 名 保 禄 · 若 瑟 （ ） ， 是 越 南 罗 马 天 主 教 枢 机 。 1963 年 被 任 为 主 教 ； 1990 年 被 擢 升 为 天 主 教 河 内 总 教 区 宗 座 署 理 ； 1994 年 被 擢 升 为 总 主 教 ， 同 年 年 底 被 擢 升 为 枢 机 ； 2009 年 2 月 离 世 。 范 廷 颂 于 1919 年 6 月 15 日 在 越 南 宁 平 省 天 主 教 发 艳 教 区 出 生 ； 童 年 时 接 受 良 好 教 育 后 ， 被 一 位 越 南 神 父 带 到 河 内 继 续 其 学 业 。 范 廷 颂 于 1940 年 在 河 内 大 修 道 院 完 成 神 学 学 业 。 范 廷 颂 于 1949 年 6 月 6 日 在 河 内 的 主 教 座 堂 晋 铎 ； 及 后 被 派 到 圣 女 小 德 兰 孤 儿 院 服 务 。 1950 年 代 ， 范 廷 颂 在 河 内 堂 区 创 建 移 民 接 待 中 心 以 收 容 到 河 内 避 战 的 难 民 。 1954 年 ， 法 越 战 争 结 束 ， 越 南 民 主 共 和 国 建 都 河 内 ， 当 时 很 多 天 主 教 神 职 人 员 逃 至 越 南 的 南 方 ， 但 范 廷 颂 仍 然 留 在 河 内 。 翌 年 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 越 战 争 结 束 ， 越 南 民 主 共 和 国 建 都 河 内 ， 当 时 很 多 天 主 教 神 职 人 员 逃 至 越 南 的 南 方 ， 但 范 廷 颂 仍 然 留 在 河 内 。 翌 年 管 理 圣 若 望 小 修 院 ； 惟 在 1960 年 因 捍 卫 修 院 的 自 由 、 自 治 及 拒 绝 政 府 在 修 院 设 政 治 课 的 要 求 而 被 捕 。 1963 年 4 月 5 日 ， 教 宗 任 命 范 廷 颂 为 天 主 教 北 宁 教 区 主 教 ， 同 年 8 月 15 日 就 任 ； 其 牧 铭 为 「 我 信 天 主 的 爱 」 。 由 于 范 廷 颂 被 越 南 政 府 软 禁 差 不 多 30 年 ， 因 此 他 无 法 到 所 属 堂 区 进 行 牧 灵 工 作 而 专 注 研 读 等 工 作 。 范 廷 颂 除 了 面 对 战 争 、 贫 困 、 被 当 局 迫 害 天 主 教 会 等 问 题 外 ， 也 秘 密 恢 复 修 院 、 创 建 女 修 会 团 体 等 。 1990 年 ， 教 宗 若 望 保 禄 二 世 在 同 年 6 月 18 日 擢 升 范 廷 颂 为 天 主 教 河 内 总 教 区 宗 座 署 理 以 填 补 该 教 区 总 主 教 的 空 缺 。 1994 年 3 月 23 日 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 若 望 保 禄 二 世 在 同 年 6 月 18 日 擢 升 范 廷 颂 为 天 主 教 河 内 总 教 区 宗 座 署 理 以 填 补 该 教 区 总 主 教 的 空 缺 。 1994 年 3 月 23 日 ， 范 廷 颂 被 教 宗 若 望 保 禄 二 世 擢 升 为 天 主 教 河 内 总 教 区 总 主 教 并 兼 天 主 教 谅 山 教 区 宗 座 署 理 ； 同 年 11 月 26 日 ， 若 望 保 禄 二 世 擢 升 范 廷 颂 为 枢 机 。 范 廷 颂 在 1995 年 至 2001 年 期 间 出 任 天 主 教 越 南 主 教 团 主 席 。 2003 年 4 月 26 日 ， 教 宗 若 望 保 禄 二 世 任 命 天 主 教 谅 山 教 区 兼 天 主 教 高 平 教 区 吴 光 杰 主 教 为 天 主 教 河 内 总 教 区 署 理 主 教 ； 及 至 2005 年 2 月 19 日 ， 范 廷 颂 因 获 批 辞 去 总 主 教 职 务 而 荣 休 ； 吴 光 杰 同 日 真 除 天 主 教 河 内 总 教 区 总 主 教 职 务 。 范 廷 颂 于 2009 年 2 月 22 日 清 晨 在 河 内 离 世 ， 享 年 89 岁 ； 其 葬 礼 于 同 月 26 日 上 午 在 天 主 教 河 内 总 教 区 总 主 教 座 堂 [SEP]
[CLS] 范 廷 颂 是 什 么 时 候 被 任 为 主 教 的 ？ [SEP] 职 务 。 范 廷 颂 于 2009 年 2 月 22 日 清 晨 在 河 内 离 世 ， 享 年 89 岁 ； 其 葬 礼 于 同 月 26 日 上 午 在 天 主 教 河 内 总 教 区 总 主 教 座 堂 举 行 。 [SEP]
```

可以看到，对上下文的分块使得这个样本被切分为了 4 个新样本。

对于包含答案的样本，标签就是起始和结束 token 的索引；对于不包含答案或只有部分答案的样本，对应的标签都为 `start_position = end_position = 0`（即 `[CLS]`）。因此我们还需要设置分词器参数 `return_offsets_mapping=True`，这样就可以运用快速分词器提供的 offset mapping 映射得到对应的 token 索引。

例如我们处理前 4 个训练样本：

```python
contexts = [train_data[idx]["context"] for idx in range(4)]
questions = [train_data[idx]["question"] for idx in range(4)]

inputs = tokenizer(
    questions,
    contexts,
    max_length=300,
    truncation="only_second",
    stride=50,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print(inputs.keys())
print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
```

```
dict_keys([
  'input_ids', 
  'token_type_ids', 
  'attention_mask', 
  'offset_mapping', 
  'overflow_to_sample_mapping'
])
The 4 examples gave 14 features.
Here is where each comes from: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3].
```

由于我们设置 `return_overflowing_tokens` 和 `return_offsets_mapping`，因此编码结果中除了 input IDs、token type IDs 和 attention mask 以外，还返回了记录 token 到原文映射的 `offset_mapping`，以及记录分块样本到原始样本映射的 `overflow_to_sample_mapping`。这里 4 个样本共被分块成了 14 个新样本，其中前 4 个新样本来自于原始样本 0，接着 3 个新样本来自于样本 1 ...等等。

获得这两个映射之后，我们就可以方便地将答案文本的在原文中的起始/结束位置映射到每个块的 token 索引，以构建答案标签 `start_positions` 和 `end_positions`。这里我们简单地选择答案列表中的第一个作为参考答案：

```python
answers = [train_data[idx]["answers"] for idx in range(4)]
start_positions = []
end_positions = []

for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If the answer is not fully inside the context, label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions.append(0)
        end_positions.append(0)
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)

print(start_positions)
print(end_positions)
```

```
[47, 0, 0, 0, 53, 0, 0, 100, 0, 0, 0, 0, 61, 0]
[48, 0, 0, 0, 70, 0, 0, 124, 0, 0, 0, 0, 106, 0]
```

> **注意：**为了找到 token 序列中上下文的索引范围，我们可以直接使用 token type IDs，但是一些模型（例如 DistilBERT）的分词器并不会输出该项，因此这里使用快速分词器返回 BatchEncoding 自带的 `sequence_ids()` 函数。

下面我们做个简单的验证，例如对于第一个新样本，可以看到处理后的答案标签为 `(47, 48)`，我们将对应的 token 解码并与标注答案进行对比：

```python
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_positions[idx]
end = end_positions[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])

print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")
```

```
Theoretical answer: 1963年, labels give: 1963 年
```

> **注意：**如果使用 XLNet 等模型，padding 操作会在序列左侧进行，并且问题和上下文也会调换，`[CLS]` 也可能不在 0 位置。

**训练批处理函数**

最后，我们合并上面的这些操作，编写对应于训练集的批处理函数。由于分块后大部分的样本长度都差不多，因此没必要再进行动态 padding，这里我们简单地将所有新样本都填充到设置的最大长度。

```python
from torch.utils.data import DataLoader

max_length = 384
stride = 128

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return batch_data, torch.tensor(start_positions), torch.tensor(end_positions)
 
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=train_collote_fn)
```

我们尝试打印出一个 batch 的数据，以验证是否处理正确，并且计算分块后新数据集的大小：

```python
import torch

batch_X, batch_Start, batch_End = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_Start shape:', batch_Start.shape)
print('batch_End shape:', batch_End.shape)
print(batch_X)
print(batch_Start)
print(batch_End)

print('train set size: ', )
print(len(train_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in train_dataloader]))
```

```
batch_X shape: {
    'input_ids': torch.Size([8, 384]), 
    'token_type_ids': torch.Size([8, 384]), 
    'attention_mask': torch.Size([8, 384])
}
batch_Start shape: torch.Size([8])
batch_End shape: torch.Size([8])

{'input_ids': tensor([
        [ 101,  100, 6858,  ...,    0,    0,    0],
        [ 101,  784,  720,  ..., 1184, 7481,  102],
        [ 101,  784,  720,  ..., 3341, 8024,  102],
        ...,
        [ 101, 7716, 5335,  ...,    0,    0,    0],
        [ 101, 1367, 7063,  ..., 5638, 1867,  102],
        [ 101, 1367, 7063,  ...,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 1, 1, 1],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 1, 1],
        [0, 0, 0,  ..., 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]])
}
tensor([ 98,  10,   0,   0,  62,   0, 132,   0])
tensor([100,  35,   0,   0,  65,   0, 140,   0])

train set size: 
10142 -> 18960
```

可以看到，DataLoader 按照我们设置的 `batch_size=4` 对样本进行编码，并且成功生成了分别对应答案起始/结束索引的答案标签 `start_positions` 和 `end_positions` 。经过分块操作后，4 个原始样本被切分成了 8 个新样本，整个训练集的大小从 10142 增长到了 18960。

> 分块操作使得每一个 batch 处理后的大小参差不齐，每次送入模型的样本数并不一致，这虽然可以正常训练，但可能会影响模型最终的精度。更好地方式是为分块后的新样本重新建立一个 Dataset，然后按批加载新的数据集：
>
> ```python
> from transformers import default_data_collator
> 
> train_dataloader = DataLoader(
>     new_train_dataset,
>     shuffle=True,
>     collate_fn=default_data_collator,
>     batch_size=8,
> )
> ```

**验证/测试批处理函数**

对于验证/测试集，我们关注的不是预测出的标签序列，而是最终的答案文本，这就需要：

1. 记录每个原始样本被分块成了哪几个新样本，从而合并对应的预测结果；
2. 在 offset mapping 中标记问题的对应 token，从而在后处理阶段可以区分哪些位置的 token 来自于上下文。

因此，对应于验证集/测试集的批处理函数为：

```python
def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping').numpy().tolist()
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_mapping[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return batch_data, offset_mapping, example_ids

valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=test_collote_fn)
```

同样地，我们打印出一个 batch 编码后的数据，并且计算分块后新数据集的大小：

```python
batch_X, offset_mapping, example_ids = next(iter(valid_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print(example_ids)

print('valid set size: ')
print(len(valid_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in valid_dataloader]))
```

```
batch_X shape: {
    'input_ids': torch.Size([16, 384]), 
    'token_type_ids': torch.Size([16, 384]), 
    'attention_mask': torch.Size([16, 384])
}

['DEV_0_QUERY_0', 'DEV_0_QUERY_0', 'DEV_0_QUERY_1', 'DEV_0_QUERY_1', 'DEV_0_QUERY_2', 'DEV_0_QUERY_2', 'DEV_1_QUERY_0', 'DEV_1_QUERY_0', 'DEV_1_QUERY_1', 'DEV_1_QUERY_1', 'DEV_1_QUERY_2', 'DEV_1_QUERY_2', 'DEV_1_QUERY_3', 'DEV_1_QUERY_3', 'DEV_2_QUERY_0', 'DEV_2_QUERY_0']

valid set size: 
3219 -> 6254
```

可以看到，我们成功构建了记录每个分块对应样本 ID 的 `example_id`。经过分块操作后，整个测试集的样本数量从 3219 增长到了 6254。

至此，数据预处理部分就全部完成了！

## 2. 训练模型

对于抽取式问答任务，可以直接使用 Transformers 库自带的 `AutoModelForQuestionAnswering` 函数来构建模型。考虑到这种方式不够灵活，因此与[序列标注任务](/2022/03/18/transformers-note-6.html)一样，本文采用继承 Transformers 库预训练模型的方式来手工构建模型：

```python
from torch import nn
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class BertForExtractiveQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits
    
config = AutoConfig.from_pretrained(checkpoint)
model = BertForExtractiveQA.from_pretrained(checkpoint, config=config).to(device)
print(model)
```

```
BertForExtractiveQA(
  (bert): BertModel(...)
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

可以看到，我们构建的模型首先运用 BERT 模型将每一个 token 都编码为语义向量，然后将输出序列送入到一个包含 2 个神经元的线性全连接层中，分别表示每个 token 为答案起始、结束位置的分数，最后我们通过 `tensor.split()` 函数把输出拆分为起始、结束位置的预测值。

为了测试模型的操作是否符合预期，我们尝试将一个 batch 的数据送入模型：

```python
seed_everything(5)
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=train_collote_fn)

batch_X, _, _ = next(iter(train_dataloader))
start_outputs, end_outputs = model(batch_X)
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('start_outputs shape', start_outputs.shape)
print('end_outputs shape', end_outputs.shape)
```

```
batch_X shape: {
    'input_ids': torch.Size([8, 384]), 
    'token_type_ids': torch.Size([8, 384]), 
    'attention_mask': torch.Size([8, 384])
}
start_outputs shape torch.Size([8, 384])
end_outputs shape torch.Size([8, 384])
```

对于 batch 内 8 个都被填充到长度为 384 的文本块，模型对每个 token 都应该输出 1 个 logits 值，对应该 token 为答案起始/结束位置的分数，因此这里模型的起始/结束输出尺寸 $8\times 384$ 完全符合预期。

### 训练循环

与之前一样，我们将每一轮 Epoch 分为“训练循环”和“验证/测试循环”，在训练循环中计算损失、优化模型参数，在验证/测试循环中评估模型性能。下面我们首先实现训练循环。

如果换一个角度，我们判断每个 token 是否为答案的起始/结束位置，其实就是在整个序列所有的 $L$ 个 token 上选出一个 token 作为答案的起始/结束，相当是在进行一个 $L$ 分类问题。因此这里我们分别在起始和结束的输出上运用交叉熵来计算损失，然后取两个损失的平均值作为模型的整体损失：

```python
from tqdm.auto import tqdm

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, start_pos, end_pos) in enumerate(dataloader, start=1):
        X, start_pos, end_pos = X.to(device), start_pos.to(device), end_pos.to(device)
        start_pred, end_pred = model(X)
        start_loss = loss_fn(start_pred, start_pos)
        end_loss = loss_fn(end_pred, end_pos)
        loss = (start_loss + end_loss) / 2

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

因为最终是根据预测出的答案文本来评估模型的性能，所以在编写“验证/测试循环”之前，我们先讨论一下抽取式问答模型的后处理——怎么将模型的输出转换为答案文本。

之前在[快速分词器](/2022/03/08/transformers-note-5.html#3-抽取式问答任务)章节中已经介绍过，对每个样本，问答模型都会输出两个张量，分别对应答案起始/结束位置的 logits 值，我们回顾一下之前的后处理过程：

1. 遮盖掉除上下文之外的其他 token 的起始/结束 logits 值；

2. 通过 softmax 函数将起始/结束 logits 值转换为概率值；

3. 通过计算概率值的乘积估计每一对 `(start_token, end_token)` 为答案的分数；

4. 输出合理的（例如 `start_token` 要小于 `end_token`）分数最大的对作为答案。

本文我们会稍微做一些调整：

- 首先，我们只关心答案文本并不关心其概率，因此这里跳过 softmax 函数，直接基于 logits 值来估计答案分数，这样就从原来计算概率值的乘积变成计算 logits 值的和（因为 $\log(ab) = \log(a) + \log(b)$）；
- 其次，为了减少计算量，我们不再为所有可能的 `(start_token, end_token)` 对打分，而是只计算 logits 值最高的前 n_best 个 token  组成的对。

由于我们的 BERT 模型还没有进行微调，因此这里我们选择一个已经预训练好的问答模型 [Chinese RoBERTa-Base Model for QA](https://huggingface.co/uer/roberta-base-chinese-extractive-qa) 进行演示，对验证集上的前 12 个样本进行处理：

```python
valid_data = CMRC2018('data/cmrc2018/cmrc2018_dev.json')
small_eval_set = [valid_data[idx] for idx in range(12)]

trained_checkpoint = "uer/roberta-base-chinese-extractive-qa"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = DataLoader(small_eval_set, batch_size=4, shuffle=False, collate_fn=test_collote_fn)

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import AutoModelForQuestionAnswering
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)
```

接下来，与之前任务中的验证/测试循环一样，在 `torch.no_grad()` 上下文管理器下，使用模型对所有分块后的新样本进行预测，并且汇总预测出的起始/结束 logits 值：

```python
start_logits = []
end_logits = []

trained_model.eval()
for batch_data, _, _ in eval_set:
    batch_data = batch_data.to(device)
    with torch.no_grad():
        outputs = trained_model(**batch_data)
    start_logits.append(outputs.start_logits.cpu().numpy())
    end_logits.append(outputs.end_logits.cpu().numpy())

import numpy as np
start_logits = np.concatenate(start_logits)
end_logits = np.concatenate(end_logits)
```

在将预测结果转换为文本之前，我们还需要知道每个样本被分块为了哪几个新样本，从而汇总对应的预测结果，因此下面先构造一个记录样本 ID 到新样本索引的映射：

```python
all_example_ids = []
all_offset_mapping = []
for _, offset_mapping, example_ids in eval_set:
    all_example_ids += example_ids
    all_offset_mapping += offset_mapping

import collections
example_to_features = collections.defaultdict(list)
for idx, feature_id in enumerate(all_example_ids):
    example_to_features[feature_id].append(idx)

print(example_to_features)
```

```
defaultdict(<class 'list'>, {
    'DEV_0_QUERY_0': [0, 1], 'DEV_0_QUERY_1': [2, 3], 'DEV_0_QUERY_2': [4, 5], 'DEV_1_QUERY_0': [6, 7], 
    'DEV_1_QUERY_1': [8, 9], 'DEV_1_QUERY_2': [10, 11], 'DEV_1_QUERY_3': [12, 13], 'DEV_2_QUERY_0': [14, 15], 
    'DEV_2_QUERY_1': [16, 17], 'DEV_2_QUERY_2': [18, 19], 'DEV_3_QUERY_0': [20], 'DEV_3_QUERY_1': [21]
})
```

接下来我们只需要遍历数据集中的样本，首先汇总由其分块出的新样本的预测结果，然后取出每个新样本最高的前 `n_best` 个起始/结束 logits 值，最后评估对应的 token 片段为答案的分数（这里我们还通过限制答案的最大长度来进一步减小计算量）：

```python
n_best = 20
max_answer_length = 30
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = all_offset_mapping[feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if (end_index < start_index or end_index - start_index + 1 > max_answer_length):
                    continue
                answers.append(
                    {
                        "start": offsets[start_index][0],
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append({
            "id": example_id, 
            "prediction_text": best_answer["text"], 
            "answer_start": best_answer["start"]
        })
    else:
        predicted_answers.append({
            "id": example_id, 
            "prediction_text": "", 
            "answer_start": 0
        })
```

下面我们同步打印出预测和标注的答案来进行对比：

```python
for pred, label in zip(predicted_answers, theoretical_answers):
    print(pred['id'])
    print('pred:', pred['prediction_text'])
    print('label:', label['answers']['text'])
```

```
DEV_0_QUERY_0
pred: 光荣和ω-force
label: ['光荣和ω-force', '光荣和ω-force', '光荣和ω-force']
DEV_0_QUERY_1
pred: 任天堂游戏谜之村雨城
label: ['村雨城', '村雨城', '任天堂游戏谜之村雨城']
...
```

可以看到，由于 [Chinese RoBERTa-Base Model for QA](https://huggingface.co/uer/roberta-base-chinese-extractive-qa) 模型本身的预训练数据就包含了 [CMRC 2018](https://ymcui.com/cmrc2018/)，因此模型的预测结果非常好。

在成功获取到预测的答案片段之后，就可以对模型的性能进行评估了。这里我们对 CMRC 2018 自带的[评估脚本](https://github.com/ymcui/cmrc2018/blob/master/squad-style-data/cmrc2018_evaluate.py)进行修改，使其支持本文模型的输出格式。请将下面的代码存放在 *cmrc2018_evaluate.py* 文件中，后续直接使用其中的 `evaluate` 函数进行评估。

```python
import re
import sys
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenize = lambda x: tokenizer(x).tokens()[1:-1]

# import nltk
# tokenize = lambda x: nltk.word_tokenize(x)

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                # ss = nltk.word_tokenize(temp_str)
                ss = tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        # ss = nltk.word_tokenize(temp_str)
        ss = tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax

def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision     = 1.0*lcs_len/len(prediction_segs)
        recall         = 1.0*lcs_len/len(ans_segs)
        f1             = (2*precision*recall)/(precision+recall)
        f1_scores.append(f1)
    return max(f1_scores)

def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

def evaluate(predictions, references):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    pred = dict([(data['id'], data['prediction_text']) for data in predictions])
    ref = dict([(data['id'], data['answers']['text']) for data in references])
    for query_id, answers in ref.items():
        total_count += 1
        if query_id not in pred:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue
        prediction = pred[query_id]
        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)
    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {
        'avg': (em_score + f1_score) * 0.5, 
        'f1': f1_score, 
        'em': em_score, 
        'total': total_count, 
        'skip': skip_count
    }
```

最后，我们将上面的预测结果送入 `evaluate` 函数进行评估：

```python
from cmrc2018_evaluate import evaluate

result = evaluate(predicted_answers, theoretical_answers)
print(f"F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
```

```
F1: 92.63 EM: 75.00 AVG: 83.81
```

### 测试循环

熟悉了后处理操作之后，编写验证/测试循环就很简单了，只需对上面的这些步骤稍作整合即可：

```python
import collections
from cmrc2018_evaluate import evaluate

n_best = 20
max_answer_length = 30

def test_loop(dataloader, dataset, model):
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            outputs = model(**batch_data)
        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]} for s_idx in range(len(dataset))
    ]
    predicted_answers = []
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
    return result
```

为了方便后续保存验证集上最好的模型，这里我们还在验证/测试循环中返回对模型预测的评估结果。

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
best_avg_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model, mode='Valid')
    avg_score = valid_scores['avg']
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
print("Done!")
```

下面，我们正式开始训练，完整的训练代码如下：

```python
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from transformers import BertPreTrainedModel, BertModel
from transformers import AdamW, get_scheduler
import json
import collections
import sys
from tqdm.auto import tqdm
sys.path.append('./')
from cmrc2018_evaluate import evaluate

max_length = 384
stride = 128
n_best = 20
max_answer_length = 30
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

seed_everything(7)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class CMRC2018(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            idx = 0
            for article in json_data['data']:
                title = article['title']
                context = article['paragraphs'][0]['context']
                for question in article['paragraphs'][0]['qas']:
                    q_id = question['id']
                    ques = question['question']
                    text = [ans['text'] for ans in question['answers']]
                    answer_start = [ans['answer_start'] for ans in question['answers']]
                    Data[idx] = {
                        'id': q_id,
                        'title': title,
                        'context': context, 
                        'question': ques,
                        'answers': {
                            'text': text,
                            'answer_start': answer_start
                        }
                    }
                    idx += 1
        return Data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = CMRC2018('data/cmrc2018/cmrc2018_train.json')
valid_data = CMRC2018('data/cmrc2018/cmrc2018_dev.json')
test_data = CMRC2018('data/cmrc2018/cmrc2018_trial.json')

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
        batch_answers.append(sample['answers'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return batch_data, torch.tensor(start_positions), torch.tensor(end_positions)

def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping').numpy().tolist()
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_mapping[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return batch_data, offset_mapping, example_ids

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_collote_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=test_collote_fn)

print('train set size: ', )
print(len(train_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in train_dataloader]))
print('valid set size: ')
print(len(valid_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in valid_dataloader]))
print('test set size: ')
print(len(test_data), '->', sum([batch_data['input_ids'].shape[0] for batch_data, _, _ in test_dataloader]))

class BertForExtractiveQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits

config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = 2
model = BertForExtractiveQA.from_pretrained(checkpoint, config=config).to(device)

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1) * len(dataloader)
    
    model.train()
    for batch, (X, start_pos, end_pos) in enumerate(dataloader, start=1):
        X, start_pos, end_pos = X.to(device), start_pos.to(device), end_pos.to(device)
        start_pred, end_pred = model(X)
        start_loss = loss_fn(start_pred, start_pos)
        end_loss = loss_fn(end_pred, end_pos)
        loss = (start_loss + end_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, dataset, model):
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            pred_start_logits, pred_end_logit = model(batch_data)
        start_logits.append(pred_start_logits.cpu().numpy())
        end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    theoretical_answers = [
        {"id": dataset[s_idx]["id"], "answers": dataset[s_idx]["answers"]} for s_idx in range(len(dataset))
    ]
    predicted_answers = []
    for s_idx in tqdm(range(len(dataset))):
        example_id = dataset[s_idx]["id"]
        context = dataset[s_idx]["context"]
        answers = []
        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
    result = evaluate(predicted_answers, theoretical_answers)
    print(f"F1: {result['f1']:>0.2f} EM: {result['em']:>0.2f} AVG: {result['avg']:>0.2f}\n")
    return result

loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)

total_loss = 0.
best_avg_score = 0.
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_scores = test_loop(valid_dataloader, valid_data, model)
    avg_score = valid_scores['avg']
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_avg_{avg_score:0.4f}_model_weights.bin')
print("Done!")
```

```
Using cuda device

train set size: 
10142 -> 18960
valid set size: 
3219 -> 6254
test set size: 
1002 -> 1961

Epoch 1/3
-------------------------------
loss: 1.473110: 100%|█████████████| 2536/2536 [08:13<00:00,  5.14it/s]
100%|█████████████████████████████| 805/805 [00:50<00:00, 15.86it/s]
100%|█████████████████████████████| 3219/3219 [00:00<00:00, 4216.22it/s]
F1: 82.96 EM: 63.84 AVG: 73.40

saving new weights...

Epoch 2/3
-------------------------------
loss: 1.178375: 100%|█████████████| 2536/2536 [08:13<00:00,  5.14it/s]
100%|█████████████████████████████| 805/805 [00:51<00:00, 15.78it/s]
100%|█████████████████████████████| 3219/3219 [00:00<00:00, 4375.79it/s]
F1: 84.97 EM: 65.52 AVG: 75.24

saving new weights...

Epoch 3/3
-------------------------------
loss: 1.010483: 100%|█████████████| 2536/2536 [08:13<00:00,  5.14it/s]
100%|█████████████████████████████| 805/805 [00:51<00:00, 15.77it/s]
100%|█████████████████████████████| 3219/3219 [00:00<00:00, 4254.01it/s]
F1: 83.84 EM: 63.40 AVG: 73.62

Done!
```

可以看到，随着训练的进行，模型在验证集上的性能先升后降。因此，3 轮训练结束后，目录下只保存了前两轮训练后的模型权重：

```
epoch_1_valid_avg_73.3993_model_weights.bin
epoch_2_valid_avg_75.2441_model_weights.bin
```

至此，我们对 BERT 摘要模型的训练就完成了。

## 3. 测试模型

训练完成后，我们加载在验证集上性能最优的模型权重，汇报其在测试集上的性能，并且将模型的预测结果保存到文件中。

```python
model.load_state_dict(torch.load('epoch_2_valid_avg_75.2441_model_weights.bin'))

model.eval()
with torch.no_grad():
    print('evaluating on test set...')
    all_example_ids = []
    all_offset_mapping = []
    for _, offset_mapping, example_ids in test_dataloader:
        all_example_ids += example_ids
        all_offset_mapping += offset_mapping
    example_to_features = collections.defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    start_logits = []
    end_logits = []
    model.eval()
    for batch_data, _, _ in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        pred_start_logits, pred_end_logit = model(batch_data)
        start_logits.append(pred_start_logits.cpu().numpy())
        end_logits.append(pred_end_logit.cpu().numpy())
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    
    theoretical_answers = [
        {"id": test_data[s_idx]["id"], "answers": test_data[s_idx]["answers"]} for s_idx in range(len(test_dataloader))
    ]
    predicted_answers = []
    save_resluts = []
    for s_idx in tqdm(range(len(test_data))):
        example_id = test_data[s_idx]["id"]
        context = test_data[s_idx]["context"]
        title = test_data[s_idx]["title"]
        question = test_data[s_idx]["question"]
        labels = test_data[s_idx]["answers"]
        answers = []
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = all_offset_mapping[feature_index]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (end_index < start_index or end_index-start_index+1 > max_answer_length):
                        continue
                    answers.append({
                        "start": offsets[start_index][0], 
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], 
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    })
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
            save_resluts.append({
                "id": example_id, 
                "title": title, 
                "context": context, 
                "question": question, 
                "answers": labels, 
                "prediction_text": best_answer["text"], 
                "answer_start": best_answer["start"]
            })
        else:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "", 
                "answer_start": 0
            })
            save_resluts.append({
                "id": example_id, 
                "title": title, 
                "context": context, 
                "question": question, 
                "answers": labels, 
                "prediction_text": "", 
                "answer_start": 0
            })
    eval_result = evaluate(predicted_answers, theoretical_answers)
    print(f"F1: {eval_result['f1']:>0.2f} EM: {eval_result['em']:>0.2f} AVG: {eval_result['avg']:>0.2f}\n")
    print('saving predicted results...')
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in save_resluts:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')
```

```
evaluating on test set...
100%|█████████████████████████████| 251/251 [00:15<00:00, 15.95it/s]
100%|█████████████████████████████| 1002/1002 [00:00<00:00, 3243.72it/s]
F1: 69.10 EM: 31.47 AVG: 50.29

saving predicted results...
```

可以看到，最终问答模型在测试集上取得了 F1 值 69.10、EM 值 31.47 的结果。考虑到我们只使用了基础版本的 BERT 模型，并且只训练了 3 轮，这已经是一个不错的结果了。

我们打开保存预测结果的 *test_data_pred.json*，其中每一行对应一个样本，`sentence` 对应原文，`pred_label` 对应预测出的实体，`true_label` 对应标注实体信息。

```
{
  "id": "TRIAL_800_QUERY_0", 
  "title": "泡泡战士", 
  "context": "基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师) ，可选择经典、热血、狙击等模式进行游戏。若游戏中离，则4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。", 
  "question": "生命数耗完即算为什么？", 
  "answers": {
    "text": ["踢爆"], 
    "answer_start": [127]
  }, 
  "prediction_text": "踢爆", 
  "answer_start": 182
}
...
```

至此，我们使用 Transformers 库进行抽取式问答任务就全部完成了！

## 代码

与之前一样，我们按照功能将代码拆分成模块并且存放在不同的文件中，整理后的代码存储在 Github：
[How-to-use-Transformers/src/sequence_labeling_extractiveQA_cmrc/](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/sequence_labeling_extractiveQA_cmrc)

与 Transformers 库类似，我们将模型损失的计算也包含进模型本身，这样在训练循环中我们就可以直接使用模型返回的损失进行反向传播。

运行 *run_extractiveQA.sh* 脚本即可进行训练。如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的 F1 和 EM 值分别为  67.96 和 31.84 （Nvidia Tesla V100, batch=4）。

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://pytorch.org/docs/stable/) Pytorch 官方文档  
[[3]](https://huggingface.co/docs/transformers/index) Transformers 官方文档