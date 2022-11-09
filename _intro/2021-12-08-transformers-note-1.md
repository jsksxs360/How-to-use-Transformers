---
title: 第四章：开箱即用的 pipelines
author: SHENG XU
date: 2021-12-08
category: NLP
layout: post
---

通过前三章的介绍，相信你已经对自然语言处理 (NLP) 以及 Transformer 模型有了一定的了解。从本章开始将正式进入正题——Transformers 库的组件以及使用方法。

本章将通过一些封装好的 pipelines 向大家展示 Transformers 库的强大能力。

## 开箱即用的 pipelines

Transformers 库将目前的 NLP 任务归纳为几下几类：

- **文本分类：**例如情感分析、句子对关系判断等；
- **对文本中的词语进行分类：**例如词性标注 (POS)、命名实体识别 (NER) 等；
- **文本生成：**例如填充预设的模板 (prompt)、预测文本中被遮掩掉 (masked) 的词语；
- **从文本中抽取答案：**例如根据给定的问题从一段文本中抽取出对应的答案；
- **根据输入文本生成新的句子：**例如文本翻译、自动摘要等。

Transformers 库最基础的对象就是 `pipeline()` 函数，它封装了预训练模型和对应的前处理和后处理环节。我们只需输入文本，就能得到预期的答案。目前常用的 [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) 有：

- `feature-extraction` （获得文本的向量化表示）
- `fill-mask` （填充被遮盖的词、片段）
- `ner`（命名实体识别）
- `question-answering` （自动问答）
- `sentiment-analysis` （情感分析）
- `summarization` （自动摘要）
- `text-generation` （文本生成）
- `translation` （机器翻译）
- `zero-shot-classification` （零训练样本分类）

下面我们以常见的几个 NLP 任务为例，展示如何调用这些 pipeline 模型。

### 情感分析

借助情感分析 pipeline，我们只需要输入文本，就可以得到其情感标签（积极/消极）以及对应的概率：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
results = classifier(
  ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)
```

```
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

[{'label': 'POSITIVE', 'score': 0.9598048329353333}]
[{'label': 'POSITIVE', 'score': 0.9598048329353333}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
```

pipeline 模型会自动完成以下三个步骤：

1. 将文本预处理为模型可以理解的格式；
2. 将预处理好的文本送入模型；
3. 对模型的预测值进行后处理，输出人类可以理解的格式。

pipeline 会自动选择合适的预训练模型来完成任务。例如对于情感分析，默认就会选择微调好的英文情感模型 *distilbert-base-uncased-finetuned-sst-2-english*。

> Transformers 库会在创建对象时下载并且缓存模型，只有在首次加载模型时才会下载，后续会直接调用缓存好的模型。

### 零训练样本分类

零训练样本分类 pipeline 允许我们在不提供任何标注数据的情况下自定义分类标签。

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"This is a course about the Transformers library",
candidate_labels=["education", "politics", "business"],
)
print(result)
```

```
No model was supplied, defaulted to facebook/bart-large-mnli (https://huggingface.co/facebook/bart-large-mnli)

{'sequence': 'This is a course about the Transformers library', 
 'labels': ['education', 'business', 'politics'], 
 'scores': [0.8445973992347717, 0.11197526752948761, 0.043427325785160065]}
```

可以看到，pipeline 自动选择了预训练好的 *facebook/bart-large-mnli* 模型来完成任务。

### 文本生成

我们首先根据任务需要构建一个模板 (prompt)，然后将其送入到模型中来生成后续文本。注意，由于文本生成具有随机性，因此每次运行都会得到不同的结果。

> 这种模板被称为前缀模板 (Preﬁx Prompt)，了解更多详细信息可以查看[《Prompt 方法简介》](/2022/09/10/what-is-prompt.html)。

```python
from transformers import pipeline

generator = pipeline("text-generation")
results = generator("In this course, we will teach you how to")
print(results)
results = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=50
) 
print(results)
```

```
No model was supplied, defaulted to gpt2 (https://huggingface.co/gpt2)

[{'generated_text': "In this course, we will teach you how to use data and models that can be applied in any real-world, everyday situation. In most cases, the following will work better than other courses I've offered for an undergrad or student. In order"}]
[{'generated_text': 'In this course, we will teach you how to make your own unique game called "Mono" from scratch by doing a game engine, a framework and the entire process starting with your initial project. We are planning to make some basic gameplay scenarios and'}, {'generated_text': 'In this course, we will teach you how to build a modular computer, how to run it on a modern Windows machine, how to install packages, and how to debug and debug systems. We will cover virtualization and virtualization without a programmer,'}]
```

可以看到，pipeline 自动选择了预训练好的 *gpt2* 模型来完成任务。我们也可以指定要使用的模型。对于文本生成任务，我们可以在 [Model Hub](https://huggingface.co/models) 页面左边选择 [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) tag 查询支持的模型。例如，我们在相同的 pipeline 中加载 [distilgpt2](https://huggingface.co/distilgpt2) 模型：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(results)
```

```
[{'generated_text': 'In this course, we will teach you how to use React in any form, and how to use React without having to worry about your React dependencies because'}, 
 {'generated_text': 'In this course, we will teach you how to use a computer system in order to create a working computer. It will tell you how you can use'}]
```

还可以通过左边的语言 tag 选择其他语言的模型。例如加载专门用于生成中文古诗的 [gpt2-chinese-poem](https://huggingface.co/uer/gpt2-chinese-poem) 模型：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 万 叠 春 山 积 雨 晴 ，",
    max_length=40,
    num_return_sequences=2,
)
print(results)
```

```
[{'generated_text': '[CLS] 万 叠 春 山 积 雨 晴 ， 孤 舟 遥 送 子 陵 行 。 别 情 共 叹 孤 帆 远 ， 交 谊 深 怜 一 座 倾 。 白 日 风 波 身 外 幻'}, 
 {'generated_text': '[CLS] 万 叠 春 山 积 雨 晴 ， 满 川 烟 草 踏 青 行 。 何 人 唤 起 伤 春 思 ， 江 畔 画 船 双 橹 声 。 桃 花 带 雨 弄 晴 光'}]
```

### 遮盖词填充

给定一段部分词语被遮盖掉 (masked) 的文本，使用预训练模型来预测能够填充这些位置的词语。

> 与前面介绍的文本生成类似，这个任务其实也是先构建模板然后运用模型来完善模板，称为填充模板 (Cloze Prompt)。了解更多详细信息可以查看[《Prompt 方法简介》](/2022/09/10/what-is-prompt.html)。

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
results = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(results)
```

```
No model was supplied, defaulted to distilroberta-base (https://huggingface.co/distilroberta-base)

[{'sequence': 'This course will teach you all about mathematical models.', 
  'score': 0.19619858264923096, 
  'token': 30412, 
  'token_str': ' mathematical'}, 
 {'sequence': 'This course will teach you all about computational models.', 
  'score': 0.04052719101309776, 
  'token': 38163, 
  'token_str': ' computational'}]
```

可以看到，pipeline 自动选择了预训练好的 *distilroberta-base* 模型来完成任务。

### 命名实体识别

命名实体识别 (NER) pipeline 负责从文本中抽取出指定类型的实体，例如人物、地点、组织等等。

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(results)
```

```
No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english)

[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
 {'entity_group': 'ORG', 'score': 0.97960186, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
 {'entity_group': 'LOC', 'score': 0.99321055, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

可以看到，模型正确地识别出了 Sylvain 是一个人物，Hugging Face 是一个组织，Brooklyn 是一个地名。

> 这里通过设置参数 `grouped_entities=True`，使得 pipeline 自动合并属于同一个实体的多个子词 (token)，例如这里将“Hugging”和“Face”合并为一个组织实体，实际上 Sylvain 也进行了子词合并，因为分词器会将 Sylvain 切分为 `S`、`##yl` 、`##va` 和 `##in` 四个 token。

### 自动问答

自动问答 pipeline 可以根据给定的上下文回答问题，例如：

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
answer = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(answer)
```

```
No model was supplied, defaulted to distilbert-base-cased-distilled-squad (https://huggingface.co/distilbert-base-cased-distilled-squad)

{'score': 0.6949771046638489, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

可以看到，pipeline 自动选择了在 SQuAD 数据集上训练好的 *distilbert-base* 模型来完成任务。这里的自动问答 pipeline 实际上是一个抽取式问答模型，即从给定的上下文中抽取答案，而不是生成答案。

> 根据形式的不同，自动问答 (QA) 系统可以分为三种：
>
> - **抽取式 QA (extractive QA)：**假设答案就包含在文档中，因此直接从文档中抽取答案；
> - **多选 QA (multiple-choice QA)：**从多个给定的选项中选择答案，相当于做阅读理解题；
> - **无约束 QA (free-form QA)：**直接生成答案文本，并且对答案文本格式没有任何限制。

### 自动摘要

自动摘要 pipeline 旨在将长文本压缩成短文本，并且还要尽可能保留原文的主要信息，例如：

```python
from transformers import pipeline

summarizer = pipeline("summarization")
results = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
    """
)
print(results)
```

```
No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 (https://huggingface.co/sshleifer/distilbart-cnn-12-6)

[{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering . Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance engineering .'}]
```

可以看到，pipeline 自动选择了预训练好的 *distilbart-cnn-12-6* 模型来完成任务。与文本生成类似，我们也可以通过 `max_length` 或 `min_length` 参数来控制返回摘要的长度。

## 这些 pipeline 背后做了什么？

这些简单易用的 pipeline 模型实际上封装了许多操作，下面我们就来了解一下它们背后究竟做了啥。以第一个情感分析 pipeline 为例，我们运行下面的代码

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
```

就会得到结果：

```
[{'label': 'POSITIVE', 'score': 0.9598048329353333}]
```

实际上它的背后经过了三个步骤：

1. 预处理 (preprocessing)，将原始文本转换为模型可以接受的输入格式；
2. 将处理好的输入送入模型；
3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。

<img src="/How-to-use-Transformers/assets/img/transformers-note-1/full_nlp_pipeline.png" alt="full_nlp_pipeline" style="display: block; margin: auto; width: 800px">

### 使用分词器进行预处理

因为神经网络模型无法直接处理文本，因此首先需要通过**预处理**环节将文本转换为模型可以理解的数字。具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行：

1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 **tokens**；
2. 根据模型的词表将每个 token 映射到对应的 token 编号（就是一个数字）；
3. 根据模型的需要，添加一些额外的输入。

我们对输入文本的预处理需要与模型自身预训练时的操作完全一致，只有这样模型才可以正常地工作。注意，每个模型都有特定的预处理操作，如果对要使用的模型不熟悉，可以通过 [Model Hub](https://huggingface.co/models) 查询。这里我们使用 `AutoTokenizer` 类和它的 `from_pretrained()`  函数，它可以自动根据模型 checkpoint 名称来获取对应的分词器。

情感分析 pipeline 的默认 checkpoint 是 [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)，下面我们手工下载并调用其分词器：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

可以看到，输出中包含两个键 `input_ids` 和 `attention_mask`，其中 `input_ids` 对应分词之后的 tokens 映射到的数字编号列表，而 `attention_mask` 则是用来标记哪些 tokens 是被填充的（这里“1”表示是原文，“0”表示是填充字符）。

> 先不要关注 `padding`、`truncation` 这些参数，以及 `attention_mask`  项，后面我们会详细介绍:)。

### 将预处理好的输入送入模型

预训练模型的下载方式和分词器 (tokenizer) 类似，Transformers 包提供了一个 `AutoModel` 类和对应的 `from_pretrained()` 函数。下面我们手工下载这个 distilbert-base 模型：

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

预训练模型的本体只包含基础的  Transformer 模块，对于给定的输入，它会输出一些神经元的值，称为 hidden states 或者特征 (features)。对于 NLP 模型来说，可以理解为是文本的高维语义表示。这些 hidden states 通常会被输入到其他的模型部分（称为 head），以完成特定的任务，例如送入到分类头中完成文本分类任务。

> 其实前面我们举例的所有 pipelines 都具有类似的模型结构，只是模型的最后一部分会使用不同的 head 以完成对应的任务。
>
> <img src="/How-to-use-Transformers/assets/img/transformers-note-1/transformer_and_head.png" alt="transformer_and_head" style="display: block; margin: auto; width: 800px">
>
> Transformers 库封装了很多不同的结构，常见的有：
>
> - `*Model` （返回 hidden states）
> - `*ForCausalLM` （用于条件语言模型）
> - `*ForMaskedLM` （用于遮盖语言模型）
> - `*ForMultipleChoice` （用于多选任务）
> - `*ForQuestionAnswering` （用于自动问答任务）
> - `*ForSequenceClassification` （用于文本分类任务）
> - `*ForTokenClassification` （用于 token 分类任务，例如 NER）

Transformer 模块的输出是一个维度为 (Batch size, Sequence length, Hidden size) 的三维张量，其中 Batch size 表示每次输入的样本（文本序列）数量，即每次输入多少个句子，上例中为 2；Sequence length 表示文本序列的长度，即每个句子被分为多少个 token，上例中为 16；Hidden size 表示每一个 token 经过模型编码后的输出向量（语义表示）的维度。

> 预训练模型编码后的输出向量的维度通常都很大，例如 Bert 模型 base 版本的输出为 768 维，一些大模型的输出维度为 3072 甚至更高。

我们可以打印出这里使用的 distilbert-base 模型的输出维度：

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
```

```
torch.Size([2, 16, 768])
```

Transformers 模型的输出格式类似 `namedtuple` 或字典，可以像上面那样通过属性访问，也可以通过键（`outputs["last_hidden_state"]`），甚至索引访问（`outputs[0]`）。

对于情感分析任务，很明显我们最后需要使用的是一个文本分类 head。因此，实际上我们不会使用 `AutoModel` 类，而是使用  `AutoModelForSequenceClassification`：

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits.shape)
```

```
torch.Size([2, 2])
```

可以看到，对于 batch 中的每一个样本，模型都会输出一个两维的向量（每一维对应一个标签，positive 或 negative）。

### 对模型输出进行后处理

由于模型的输出只是一些数值，因此并不适合人类阅读。例如我们打印出上面例子的输出：

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

```
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
```

模型对第一个句子输出 $[-1.5607, 1.6123]$，对第二个句子输出 $[ 4.1692, -3.3464]$，它们并不是概率值，而是模型最后一层输出的 logits 值。要将他们转换为概率值，还需要让它们经过一个 [SoftMax](https://zh.wikipedia.org/wiki/Softmax%E5%87%BD%E6%95%B0) 层，例如：

```python
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward0>)
```

> 所有 Transformers 模型都会输出 logits 值，因为训练时的损失函数通常会自动结合激活函数（例如 SoftMax）与实际的损失函数（例如交叉熵 cross entropy）。

这样模型的预测结果就是容易理解的概率值：第一个句子 $[0.0402, 0.9598]$，第二个句子 $[0.9995, 0.0005]$。最后，为了得到对应的标签，可以读取模型 config 中提供的 id2label 属性：

```python
print(model.config.id2label)
```

```
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

于是我们可以得到最终的预测结果：

- 第一个句子: NEGATIVE: 0.0402, POSITIVE: 0.9598
- 第二个句子: NEGATIVE: 0.9995, POSITIVE: 0.0005

## 小结

在本章中我们初步介绍了如何使用 Transformers 包提供的 pipeline 对象来处理各种 NLP 任务，并且对 pipeline 背后的工作原理进行了简单的说明。

在下一章中，我们会具体介绍组成 pipeline 的两个重要组件**模型**（`Models` 类）和**分词器**（`Tokenizers` 类）的参数以及使用方式。

## 参考

[[1]](https://huggingface.co/docs/transformers/index) Transformers 官方文档  
[[2]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程
