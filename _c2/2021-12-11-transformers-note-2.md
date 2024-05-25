---
title: 第五章：模型与分词器
author: SHENG XU
date: 2021-12-11
category: NLP
layout: post
---

本章我们将介绍 Transformers 库中的两个重要组件：**模型**和**分词器**。

## 5.1 模型

除了像之前使用 `AutoModel` 根据 checkpoint 自动加载模型以外，我们也可以直接使用模型对应的 `Model` 类，例如 BERT 对应的就是 `BertModel`：

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

注意，**在大部分情况下，我们都应该使用 `AutoModel` 来加载模型。**这样如果我们想要使用另一个模型（比如把 BERT 换成 RoBERTa），只需修改 checkpoint，其他代码可以保持不变。

### 加载模型

所有存储在 HuggingFace [Model Hub](https://huggingface.co/models) 上的模型都可以通过 `Model.from_pretrained()` 来加载权重，参数可以像上面一样是 checkpoint 的名称，也可以是本地路径（预先下载的模型目录），例如：

```python
from transformers import BertModel

model = BertModel.from_pretrained("./models/bert/")
```

`Model.from_pretrained()` 会自动缓存下载的模型权重，默认保存到 *~/.cache/huggingface/transformers*，我们也可以通过 HF_HOME 环境变量自定义缓存目录。

> 由于 checkpoint 名称加载方式需要连接网络，因此在大部分情况下我们都会采用本地路径的方式加载模型。
>
> 部分模型的 Hub 页面中会包含很多文件，我们通常只需要下载模型对应的 *config.json* 和 *pytorch_model.bin*，以及分词器对应的 *tokenizer.json*、*tokenizer_config.json* 和 *vocab.txt*。
{: .block-tip}

### 保存模型

保存模型通过调用 `Model.save_pretrained()` 函数实现，例如保存加载的 BERT 模型：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/")
```

这会在保存路径下创建两个文件：

- *config.json*：模型配置文件，存储模型结构参数，例如 Transformer 层数、特征空间维度等；
- *pytorch_model.bin*：又称为 state dictionary，存储模型的权重。

简单来说，配置文件记录模型的**结构**，模型权重记录模型的**参数**，这两个文件缺一不可。我们自己保存的模型同样通过 `Model.from_pretrained()` 加载，只需要传递保存目录的路径。

## 5.2 分词器

由于神经网络模型不能直接处理文本，因此我们需要先将文本转换为数字，这个过程被称为**编码 (Encoding)**，其包含两个步骤：

1. 使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens；
2. 将所有的 token 映射到对应的 token ID。

### 分词策略

根据切分粒度的不同，分词策略可以分为以下几种：

- **按词切分 (Word-based)**

  <img src="/assets/img/transformers-note-2/word_based_tokenization.png" alt="word_based_tokenization" style="display: block; margin: auto; width: 600px">

  例如直接利用 Python 的 `split()` 函数按空格进行分词：

  ```python
  tokenized_text = "Jim Henson was a puppeteer".split()
  print(tokenized_text)
  ```

  ```
  ['Jim', 'Henson', 'was', 'a', 'puppeteer']
  ```

  这种策略的问题是会将文本中所有出现过的独立片段都作为不同的 token，从而产生巨大的词表。而实际上很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果给它们赋予不同的编号就无法表示出这种关联性。

  > 词表就是一个映射字典，负责将 token 映射到对应的 ID（从 0 开始）。神经网络模型就是通过这些 token ID 来区分每一个 token。
  {: .block-tip }
  
  当遇到不在词表中的词时，分词器会使用一个专门的 $\texttt{[UNK]}$ token 来表示它是 unknown 的。显然，如果分词结果中包含很多 $\texttt{[UNK]}$ 就意味着丢失了很多文本信息，因此一个好的分词策略，应该尽可能不出现 unknown token。

- **按字符切分 (Character-based)**

  <img src="/assets/img/transformers-note-2/character_based_tokenization.png" alt="character_based_tokenization" style="display: block; margin: auto; width: 600px">

  这种策略把文本切分为字符而不是词语，这样就只会产生一个非常小的词表，并且很少会出现词表外的 tokens。

  但是从直觉上来看，字符本身并没有太大的意义，因此将文本切分为字符之后就会变得不容易理解。这也与语言有关，例如中文字符会比拉丁字符包含更多的信息，相对影响较小。此外，这种方式切分出的 tokens 会很多，例如一个由 10 个字符组成的单词就会输出 10 个 tokens，而实际上它们只是一个词。

  因此现在广泛采用的是一种同时结合了按词切分和按字符切分的方式——按子词切分 (Subword tokenization)。

- **按子词切分 (Subword) ** 

  高频词直接保留，低频词被切分为更有意义的子词。例如 “annoyingly” 是一个低频词，可以切分为 “annoying” 和 “ly”，这两个子词不仅出现频率更高，而且词义也得以保留。下图展示了对 “Let’s do tokenization!“ 按子词切分的结果：

  <img src="/assets/img/transformers-note-2/bpe_subword.png" alt="bpe_subword" style="display: block; margin: auto; width: 600px">

  可以看到，“tokenization” 被切分为了 “token” 和 “ization”，不仅保留了语义，而且只用两个 token 就表示了一个长词。这种策略只用一个较小的词表就可以覆盖绝大部分文本，基本不会产生 unknown token。尤其对于土耳其语等黏着语，几乎所有的复杂长词都可以通过串联多个子词构成。

### 加载与保存分词器

分词器的加载与保存与模型相似，使用 `Tokenizer.from_pretrained()` 和 `Tokenizer.save_pretrained()` 函数。例如加载并保存 BERT 模型的分词器：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
```

同样地，在大部分情况下我们都应该使用 `AutoTokenizer` 来加载分词器：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
```

调用 `Tokenizer.save_pretrained()` 函数会在保存路径下创建三个文件：

- *special_tokens_map.json*：映射文件，里面包含 unknown token 等特殊字符的映射关系；
- *tokenizer_config.json*：分词器配置文件，存储构建分词器需要的参数；
- *vocab.txt*：词表，一行一个 token，行号就是对应的 token ID（从 0 开始）。

### 编码与解码文本

前面说过，文本编码 (Encoding) 过程包含两个步骤：

1. **分词：**使用分词器按某种策略将文本切分为 tokens；
2. **映射：**将 tokens 转化为对应的 token IDs。 

下面我们首先使用 BERT 分词器来对文本进行分词：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

```
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
```

可以看到，BERT 分词器采用的是子词切分策略，它会不断切分词语直到获得词表中的 token，例如 “Transformer” 会被切分为 “Trans” 和 “##former”。
（旧版本有少许差别： [revision ae1d3b](https://huggingface.co/google-bert/bert-base-cased/commit/ae1d3b2cce5ef798cab884c0e7e61e34f46bc412) 之前的结果是：`['using', 'a', 'transform', '##er', 'network', 'is', 'simple']`）

然后，我们通过 `convert_tokens_to_ids()` 将切分出的 tokens 转换为对应的 token IDs：

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```
[7993, 170, 13809, 23763, 2443, 1110, 3014]
```

还可以通过 `encode()` 函数将这两个步骤合并，并且 `encode()` 会自动添加模型需要的特殊 token，例如 BERT 分词器会分别在序列的首尾添加 $\texttt{[CLS]}$ 和 $\texttt{[SEP]}$：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
sequence_ids = tokenizer.encode(sequence)

print(sequence_ids)
```

```
[101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102]
```

其中 101 和 102 分别是 $\texttt{[CLS]}$ 和 $\texttt{[SEP]}$ 对应的 token IDs。

注意，上面这些只是为了演示。**在实际编码文本时，最常见的是直接使用分词器进行处理**，这样不仅会返回分词后的 token IDs，还包含模型需要的其他输入。例如 BERT 分词器还会自动在输入中添加 `token_type_ids` 和 `attention_mask`：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_text = tokenizer("Using a Transformer network is simple")
print(tokenized_text)
```

```
{'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

文本解码 (Decoding) 与编码相反，负责将 token IDs 转换回原来的字符串。注意，解码过程不是简单地将 token IDs 映射回 tokens，还需要合并那些被分为多个 token 的单词。下面我们通过 `decode()` 函数解码前面生成的 token IDs：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

decoded_string = tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102])
print(decoded_string)
```

```
Using a transformer network is simple
[CLS] Using a Transformer network is simple [SEP]
```

解码文本是一个重要的步骤，在进行文本生成、翻译或者摘要等 Seq2Seq (Sequence-to-Sequence) 任务时都会调用这一函数。

## 5.3 处理多段文本

现实场景中，我们往往会同时处理多段文本，而且模型也只接受批 (batch) 数据作为输入，即使只有一段文本，也需要将它组成一个只包含一个样本的 batch，例如：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
# input_ids = torch.tensor(ids), This line will fail.
input_ids = torch.tensor([ids])
print("Input IDs:\n", input_ids)

output = model(input_ids)
print("Logits:\n", output.logits)
```

```
Input IDs: 
tensor([[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607,
          2026,  2878,  2166,  1012]])
Logits: 
tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward0>)
```

这里我们通过 `[ids]` 构建了一个只包含一段文本的 batch，更常见的是送入包含多段文本的 batch：

```python
batched_ids = [ids, ids, ids, ...]
```

注意，上面的代码仅作为演示。**实际场景中，我们应该直接使用分词器对文本进行处理**，例如对于上面的例子：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print("Inputs Keys:\n", tokenized_inputs.keys())
print("\nInput IDs:\n", tokenized_inputs["input_ids"])

output = model(**tokenized_inputs)
print("\nLogits:\n", output.logits)
```

```
Inputs Keys:
 dict_keys(['input_ids', 'attention_mask'])

Input IDs:
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])

Logits:
tensor([[-1.5607,  1.6123]], grad_fn=<AddmmBackward0>)
```

可以看到，分词器输出的结果中不仅包含 token IDs（`input_ids`），还会包含模型需要的其他输入项。前面我们之所以只输入 token IDs 模型也能正常运行，是因为它自动地补全了其他的输入项，例如 `attention_mask` 等，后面我们会具体介绍。

> 由于分词器自动在序列的首尾添加了 $\texttt{[CLS]}$ 和 $\texttt{[SEP]}$ token，所以上面两个例子中模型的输出是有差异的。因为 DistilBERT 预训练时是包含 $\texttt{[CLS]}$ 和 $\texttt{[SEP]}$ 的，所以下面的例子才是正确的使用方法。
>{: .block-warning}

### Padding 操作

按批输入多段文本产生的一个直接问题就是：batch 中的文本有长有短，而输入张量必须是严格的二维矩形，维度为 $(\text{batch size}, \text{sequence length})$，即每一段文本编码后的 token IDs 数量必须一样多。例如下面的 ID 列表是无法转换为张量的：

```python
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

我们需要通过 Padding 操作，在短序列的结尾填充特殊的 padding token，使得 batch 中所有的序列都具有相同的长度，例如：

```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

模型的 padding token ID 可以通过其分词器的 `pad_token_id` 属性获得。下面我们尝试将两段文本分别以独立以及 batch 的形式送入到模型中：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
```

```
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
```

> 问题出现了，使用 padding token 填充的序列的结果竟然与其单独送入模型时不同！
>
> 这是因为模型默认会编码输入序列中的所有 token 以建模完整的上下文，因此这里会将填充的 padding token 也一同编码进去，从而生成不同的语义表示。
{: .block-danger}

因此，在进行 Padding 操作时，我们必须明确告知模型哪些 token 是我们填充的，它们不应该参与编码。这就需要使用到 Attention Mask 了，在前面的例子中相信你已经多次见过它了。

### Attention Mask

Attention Mask 是一个尺寸与 input IDs 完全相同，且仅由 0 和 1 组成的张量，0 表示对应位置的 token 是填充符，不参与计算。当然，一些特殊的模型结构也会借助 Attention Mask 来遮蔽掉指定的 tokens。

对于上面的例子，如果我们通过 `attention_mask` 标出填充的 padding token 的位置，计算结果就不会有问题了：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
batched_attention_masks = [
    [1, 1, 1],
    [1, 1, 0],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
outputs = model(
    torch.tensor(batched_ids), 
    attention_mask=torch.tensor(batched_attention_masks))
print(outputs.logits)
```

```
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
```

正如前面强调的那样，在实际使用时，我们应该直接使用分词器对文本进行处理，它不仅会向 token 序列中添加模型需要的特殊字符（例如 $\texttt{[CLS]},\texttt{[SEP]}$），还会自动生成对应的 Attention Mask。

目前大部分 Transformer 模型只能接受长度不超过 512 或 1024 的 token 序列，因此对于长序列，有以下三种处理方法：

1. 使用一个支持长文的 Transformer 模型，例如 [Longformer](https://huggingface.co/transformers/model_doc/longformer.html) 和 [LED](https://huggingface.co/transformers/model_doc/led.html)（最大长度 4096）；
2. 设定最大长度 `max_sequence_length` 以**截断**输入序列：`sequence = sequence[:max_sequence_length]`。
3. 将长文切片为短文本块 (chunk)，然后分别对每一个 chunk 编码。在后面的[快速分词器](/nlp/2022-03-08-transformers-note-5.html)中，我们会详细介绍。


### 直接使用分词器

正如前面所说，在实际使用时，我们应该直接使用分词器来完成包括分词、转换 token IDs、Padding、构建 Attention Mask、截断等操作。例如：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "So have I!"
]

model_inputs = tokenizer(sequences)
print(model_inputs)
```

```
{'input_ids': [
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], 
    [101, 2061, 2031, 1045, 999, 102]], 
 'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1]]
}
```

可以看到，分词器的输出包含了模型需要的所有输入项。对于 DistilBERT 模型，就是 input IDs（`input_ids`）和 Attention Mask（`attention_mask`）。

**Padding 操作**通过 `padding` 参数来控制：

- `padding="longest"`： 将序列填充到当前 batch 中最长序列的长度；
- `padding="max_length"`：将所有序列填充到模型能够接受的最大长度，例如 BERT 模型就是 512。

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "So have I!"
]

model_inputs = tokenizer(sequences, padding="longest")
print(model_inputs)

model_inputs = tokenizer(sequences, padding="max_length")
print(model_inputs)
```

```
{'input_ids': [
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102], 
    [101, 2061, 2031, 1045, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
}

{'input_ids': [
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, ...], 
    [101, 2061, 2031, 1045, 999, 102, 0, 0, 0, ...]], 
 'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...], 
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]]
}
```

**截断操作**通过 `truncation` 参数来控制，如果 `truncation=True`，那么大于模型最大接受长度的序列都会被截断，例如对于 BERT 模型就会截断长度超过 512 的序列。此外，也可以通过 `max_length` 参数来控制截断长度：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "So have I!"
]

model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs)
```

```
{'input_ids': [
    [101, 1045, 1005, 2310, 2042, 3403, 2005, 102], 
    [101, 2061, 2031, 1045, 999, 102]], 
 'attention_mask': [
    [1, 1, 1, 1, 1, 1, 1, 1], 
    [1, 1, 1, 1, 1, 1]]
}
```

分词器还可以通过 `return_tensors` 参数指定返回的张量格式：设为 `pt` 则返回 PyTorch 张量；`tf` 则返回 TensorFlow 张量，`np` 则返回 NumPy 数组。例如：

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "So have I!"
]

model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")
print(model_inputs)

model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
print(model_inputs)
```

```
{'input_ids': tensor([
    [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
      2607,  2026,  2878,  2166,  1012,   102],
    [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0]]), 
 'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}

{'input_ids': array([
    [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662,
     12172,  2607,  2026,  2878,  2166,  1012,   102],
    [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,
         0,     0,     0,     0,     0,     0,     0]]), 
 'attention_mask': array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
```

实际使用分词器时，我们通常会同时进行 padding 操作和截断操作，并设置返回格式为 Pytorch 张量，这样就可以直接将分词结果送入模型：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokens)
output = model(**tokens)
print(output.logits)
```

```
{'input_ids': tensor([
    [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
      2607,  2026,  2878,  2166,  1012,   102],
    [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0,
         0,     0,     0,     0,     0,     0]]), 
 'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

tensor([[-1.5607,  1.6123],
        [-3.6183,  3.9137]], grad_fn=<AddmmBackward0>)
```

在 `padding=True, truncation=True` 设置下，同一个 batch 中的序列都会 padding 到相同的长度，并且大于模型最大接受长度的序列会被自动截断。

### 编码句子对

除了对单段文本进行编码以外（batch 只是并行地编码多个单段文本），对于 BERT 等包含“句子对”预训练任务的模型，它们的分词器都支持对“句子对”进行编码，例如：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
print(tokens)
```

```
{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
```

此时分词器会使用 $\texttt{[SEP]}$ token 拼接两个句子，输出形式为“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]} \text{ sentence2 } \texttt{[SEP]}$”的 token 序列，这也是 BERT 模型预期的“句子对”输入格式。

返回结果中除了前面我们介绍过的 `input_ids` 和 `attention_mask` 之外，还包含了一个 `token_type_ids` 项，用于标记哪些 token 属于第一个句子，哪些属于第二个句子。如果我们将上面例子中的 `token_type_ids` 项与 token 序列对齐：

```
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

就可以看到第一个句子“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]}$”所有 token 的 type ID 都为 0，而第二个句子“$\text{sentence2 } \texttt{[SEP]}$”对应的 token type ID 都为 1。

> 如果我们选择其他模型，分词器的输出不一定会包含 `token_type_ids` 项（例如 DistilBERT 模型）。分词器只需保证输出格式与模型预训练时的输入一致即可。
{: .block-tip}

实际使用时，我们不需要去关注编码结果中是否包含  `token_type_ids` 项，分词器会根据 checkpoint 自动调整输出格式，例如：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence1_list = ["First sentence.", "This is the second sentence.", "Third one."]
sentence2_list = ["First sentence is short.", "The second sentence is very very very long.", "ok."]

tokens = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(tokens)
print(tokens['input_ids'].shape)
```

```
{'input_ids': tensor([
        [ 101, 2034, 6251, 1012,  102, 2034, 6251, 2003, 2460, 1012,  102,    0,
            0,    0,    0,    0,    0,    0],
        [ 101, 2023, 2003, 1996, 2117, 6251, 1012,  102, 1996, 2117, 6251, 2003,
         2200, 2200, 2200, 2146, 1012,  102],
        [ 101, 2353, 2028, 1012,  102, 7929, 1012,  102,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0]]), 
 'token_type_ids': tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
}
torch.Size([3, 18])
```

可以看到分词器成功地输出了形式为“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]} \text{ sentence2 } \texttt{[SEP]}$”的 token 序列，并且将三个序列都 padding 到了相同的长度。

## 5.4 添加 Token

实际操作中，我们还经常会遇到输入中需要包含特殊标记符的情况，例如使用 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 标记出文本中的实体。由于这些自定义 token 并不在预训练模型原来的词表中，因此直接运用分词器处理就会出现问题。

例如直接使用 BERT 分词器处理下面的句子：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
print(tokenizer.tokenize(sentence))
```

```
['two', '[', 'en', '##t', '_', 'start', ']', 'cars', '[', 'en', '##t', '_', 'end', ']', 'collided', 'in', 'a', '[', 'en', '##t', '_', 'start', ']', 'tunnel', '[', 'en', '##t', '_', 'end', ']', 'this', 'morning', '.']
```

由于分词器无法识别 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ ，因此将它们都当作未知字符处理，例如“[ENT_END]”被切分成了 `'['`, `'en'`, `'##t'`, `'_'`, `'end'`, `']'` 六个 token。

此外，一些领域的专业词汇，例如使用多个词语的缩写拼接而成的医学术语，同样也不在模型的词表中，因此也会出现上面的问题。此时我们就需要将这些新 token 添加到模型的词表中，让分词器与模型可以识别并处理这些 token。

### 添加新 token

Transformers 库提供了两种方式来添加新 token，分别是：

- **[`add_tokens()`](https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_tokens) 添加普通 token：**参数是新 token 列表，如果 token 不在词表中，就会被添加到词表的最后。

  ```python
  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
  num_added_toks = tokenizer.add_tokens(["new_token1", "my_new-token2"])
  print("We have added", num_added_toks, "tokens")
  ```

  ```
  We have added 2 tokens
  ```

  为了防止 token 已经包含在词表中，我们还可以预先对新 token 列表进行过滤：

  ```python
  new_tokens = ["new_token1", "my_new-token2"]
  new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
  tokenizer.add_tokens(list(new_tokens))
  ```

- **[`add_special_tokens()`](https://huggingface.co/docs/transformers/v4.25.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_special_tokens) 添加特殊 token：**参数是包含特殊 token 的字典，键值只能从 `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`, `additional_special_tokens` 中选择。同样地，如果 token 不在词表中，就会被添加到词表的最后。添加后，还可以通过特殊属性来访问这些 token，例如 `tokenizer.cls_token` 就指向 cls token。

  ```python
  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
  special_tokens_dict = {"cls_token": "[MY_CLS]"}
    
  num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
  print("We have added", num_added_toks, "tokens")
    
  assert tokenizer.cls_token == "[MY_CLS]"
  ```
  
  ```
  We have added 1 tokens
  ```
  
  我们也可以使用 `add_tokens()` 添加特殊 token，只需要额外设置参数 `special_tokens=True`：
  
  ```python
  checkpoint = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
  num_added_toks = tokenizer.add_tokens(["[NEW_tok1]", "[NEW_tok2]"])
  num_added_toks = tokenizer.add_tokens(["[NEW_tok3]", "[NEW_tok4]"], special_tokens=True)
    
  print("We have added", num_added_toks, "tokens")
  print(tokenizer.tokenize('[NEW_tok1] Hello [NEW_tok2] [NEW_tok3] World [NEW_tok4]!'))
  ```
  
  ```
  We have added 2 tokens
  ['[new_tok1]', 'hello', '[new_tok2]', '[NEW_tok3]', 'world', '[NEW_tok4]', '!']
  ```
  
  > 特殊 token 的标准化 (normalization) 与普通 token 有一些不同，比如不会被小写。
  >
  > 这里我们使用的是不区分大小写的 BERT 模型，因此分词后添加的普通 token $\texttt{[NEW\_tok1]}$ 和 $\texttt{[NEW\_tok2]}$ 都被处理为了小写，而添加的特殊 token $\texttt{[NEW\_tok3]}$ 和 $\texttt{[NEW\_tok4]}$ 则保持大写。
  {: .block-tip }
  

对于前面的例子，很明显实体标记符 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 属于特殊 token，因此按添加特殊 token 的方式进行：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
# num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT_START]', '[ENT_END]']})
print("We have added", num_added_toks, "tokens")

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'

print(tokenizer.tokenize(sentence))
```

```
We have added 2 tokens
['two', '[ENT_START]', 'cars', '[ENT_END]', 'collided', 'in', 'a', '[ENT_START]', 'tunnel', '[ENT_END]', 'this', 'morning', '.']
```

可以看到，分词器成功地将 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 识别为 token，并且保持大写。

### 调整 embedding 矩阵

> 向词表中添加新 token 后，必须重置模型 embedding 矩阵的大小，也就是向矩阵中添加新 token 对应的 embedding，这样模型才可以正常工作，将 token 映射到对应的 embedding。
{: .block-danger }

调整 embedding 矩阵通过 `resize_token_embeddings()` 函数来实现，例如对于前面的例子：

```python
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

print('vocabulary size:', len(tokenizer))
num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)
print("After we add", num_added_toks, "tokens")
print('vocabulary size:', len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
print(model.embeddings.word_embeddings.weight.size())

# Randomly generated matrix
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
vocabulary size: 30522
After we add 2 tokens
vocabulary size: 30524
torch.Size([30524, 768])

tensor([[-0.0325, -0.0224,  0.0044,  ..., -0.0088, -0.0078, -0.0110],
        [-0.0005, -0.0167, -0.0009,  ...,  0.0110, -0.0282, -0.0013]],
       grad_fn=<SliceBackward0>)
```

可以看到，在添加 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 之后，分词器的词表大小从 30522 增加到了 30524，模型 embedding 矩阵的大小也成功调整为了 $30524\times 768$。


> 在默认情况下，新添加 token 的 embedding 是随机初始化的。
{: .block-warning }

我们尝试打印出新添加 token 对应的 embedding（新 token 会添加在词表的末尾，因此只需打印出最后两行）。如果你多次运行上面的代码，就会发现每次打印出的 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 的 embedding 是不同的。

## 5.5 Token embedding 初始化

如果有充分的语料对模型进行微调或者继续预训练，那么将新添加 token 初始化为随机向量没什么问题。但是如果训练语料较少，甚至是只有很少语料的 few-shot learning 场景下，这种做法就存在问题。研究表明，在训练数据不够多的情况下，这些新添加 token 的 embedding 只会在初始值附近小幅波动。换句话说，即使经过训练，它们的值事实上还是随机的。

### 直接赋值

因此，在很多情况下，我们需要手工初始化新添加 token 的 embedding，这可以通过直接对 embedding 矩阵赋值来实现。例如我们将上面例子中两个新 token 的 embedding 都初始化为全零向量：

```python
import torch

with torch.no_grad():
    model.embeddings.word_embeddings.weight[-2:, :] = torch.zeros([2, model.config.hidden_size], requires_grad=True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], grad_fn=<SliceBackward0>)
```

注意，初始化 embedding 的过程并不可导，因此这里通过 `torch.no_grad()` 暂停梯度的计算。

现实场景中，更为常见的做法是使用已有 token 的 embedding 来初始化新添加 token。例如对于上面的例子，我们可以将 `[ENT_START]` 和 `[ENT_END]` 的值都初始化为“entity” token 对应的 embedding。

```python
import torch

token_id = tokenizer.convert_tokens_to_ids('entity')
token_embedding = model.embeddings.word_embeddings.weight[token_id]
print(token_id)

with torch.no_grad():
    for i in range(1, num_added_toks+1):
        model.embeddings.word_embeddings.weight[-i:, :] = token_embedding.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
9178
tensor([[-0.0039, -0.0131, -0.0946,  ..., -0.0223,  0.0107, -0.0419],
        [-0.0039, -0.0131, -0.0946,  ..., -0.0223,  0.0107, -0.0419]],
       grad_fn=<SliceBackward0>)
```

> 因为 token ID 就是 token 在 embedding 矩阵中的索引，因此这里我们直接通过 `weight[token_id]` 取出“entity”对应的 embedding。

可以看到最终结果符合我们的预期，`[ENT_START]` 和 `[ENT_END]` 被初始化为相同的 embedding。

### 初始化为已有 token 的值

更为高级的做法是根据新添加 token 的语义来进行初始化。例如将值初始化为 token 语义描述中所有 token 的平均值，假设新 token $t_i$ 的语义描述为 ${w_{i,1},w_{i,2},…,w_{i,n}}$，那么初始化 $t_i$ 的 embedding 为：
$$
\boldsymbol{E}(t_i) = \frac{1}{n}\sum_{j=1}^n \boldsymbol{E}(w_{i,j})
$$

这里 $\boldsymbol{E}$ 表示预训练模型的 embedding 矩阵。对于上面的例子，我们可以分别为 $\texttt{[ENT\_START]}$ 和 $\texttt{[ENT\_END]}$ 编写对应的描述，然后再对它们的值进行初始化：

```python
descriptions = ['start of entity', 'end of entity']

with torch.no_grad():
    for i, token in enumerate(reversed(descriptions), start=1):
        tokenized = tokenizer.tokenize(token)
        print(tokenized)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        new_embedding = model.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
        model.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])
```

```
['end', 'of', 'entity']
['start', 'of', 'entity']
tensor([[-0.0340, -0.0144, -0.0441,  ..., -0.0016,  0.0318, -0.0151],
        [-0.0060, -0.0202, -0.0312,  ..., -0.0084,  0.0193, -0.0296]],
       grad_fn=<SliceBackward0>)
```

可以看到，这里成功地将 $\texttt{[ENT\_START]}$ 的 embedding 初始化为“start”、“of”、“entity”三个 token 的平均值，将 $\texttt{[ENT\_END]}$ 初始化为“end”、“of”、“entity”的平均值。

## 参考

[[1]](https://huggingface.co/docs/transformers/index) Transformers 官方文档  
[[2]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[3]](https://www.depends-on-the-definition.com/how-to-add-new-tokens-to-huggingface-transformers/) How to add new tokens to huggingface transformers vocabulary. [Tobias Sterbak](https://www.depends-on-the-definition.com/about/)  
[[4]](https://github.com/huggingface/transformers/issues/1413) Github 讨论 Adding New Vocabulary Tokens to the Models

> 最后更新时间：2023-07-02
