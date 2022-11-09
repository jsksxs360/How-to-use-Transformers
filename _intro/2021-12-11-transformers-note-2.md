---
title: 第五章：模型与分词器
author: SHENG XU
date: 2021-12-11
category: NLP
layout: post
---

在上一章中，我们通过 Transformers 库提供的 pipeline 函数展示了 Transformers 库能够完成哪些 NLP 任务，以及这些 pipelines 背后的工作原理。

本章将深入介绍 Transformers 库中的两个重要组件：**模型**（`Models` 类）和**分词器**（`Tokenizers` 类）。

## 5.1 模型

在之前介绍 pipeline 模型时，我们使用 `AutoModel` 类根据 checkpoint 名称自动加载模型。当然，我们也可以直接使用对应的 `Model` 类。例如加载 BERT 模型（包括采用 BERT 结构的其他[模型](https://huggingface.co/models?filter=bert)）：

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

这里可以直接将 `BertModel` 替换成 `AutoModel`。

<center>在大部分情况下，我们都应该使用 <code class="language-plaintext highlighter-rouge">AutoModel</code>，编写的代码应该与 checkpoint 无关。</center>

这样如果我们想要换一个预训练模型（例如把 BERT 换成 RoBERTa），只需要切换 checkpoint，其他代码可以保持不变。

### 加载模型

通过调用 `Model.from_pretrained()` 函数可以自动加载 checkpoint 对应的模型权重 (weights)。然后，我们可以直接使用模型完成它的预训练任务，或者在新的任务上对模型权重进行微调。

> `Model.from_pretrained()` 会自动缓存下载的模型权重，默认保存到 *~/.cache/huggingface/transformers*，我们也可以通过 HF_HOME 环境变量自定义缓存目录。

所有存储在 [Model Hub](https://huggingface.co/models) 上的模型都能够通过 `Model.from_pretrained()` 加载，只需要传递对应 checkpoint 的名称。当然了，我们也可以先将模型下载下来，然后将本地路径传给 `Model.from_pretrained()`，比如加载下载好的 [Bert-base 模型](https://huggingface.co/bert-base-cased)：

```python
from transformers import BertModel

model = BertModel.from_pretrained("./models/bert/")
```

部分模型的 Hub 页面中会包含很多文件，我们通常只需要下载模型对应的 *config.json* 和 *pytorch_model.bin*，以及分词器对应的 *tokenizer.json*、*tokenizer_config.json* 和 *vocab.txt*。我们在后面会详细介绍这些文件。

### 保存模型

保存模型与加载模型类似，只需要调用 `Model.save_pretrained()` 函数。例如保存加载的 BERT 模型：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
model.save_pretrained("./models/bert-base-cased/")
```

这会在保存路径下创建两个文件：

- config.json：模型配置文件，里面包含构建模型结构的必要参数；
- pytorch_model.bin：又称为 state dictionary，包含模型的所有权重。

这两个文件缺一不可，配置文件负责记录模型的**结构**，模型权重记录模型的**参数**。我们自己保存的模型同样可以通过 `Model.from_pretrained()` 加载，只需要传递保存目录的路径。

## 5.2 分词器

因为神经网络模型不能直接处理文本，我们需要先将文本转换为模型能够处理的数字，这个过程被称为**编码 (Encoding)**：先使用分词器 (Tokenizers) 将文本按词、子词、符号切分为 tokens；然后将 tokens 映射到对应的 token 编号（token IDs）。

### 分词策略

根据切分粒度的不同，分词策略大概可以分为以下几种：

- **按词切分 (Word-based)** 规则简单，而且能产生不错的结果。

  <img src="/assets/img/transformers-note-2/word_based_tokenization.png" alt="word_based_tokenization" style="display: block; margin: auto; width: 600px">

  例如直接利用 Python 自带的 `split()` 函数按空格进行分词：

  ```python
tokenized_text = "Jim Henson was a puppeteer".split()
  print(tokenized_text)
  ```
  
  ```
['Jim', 'Henson', 'was', 'a', 'puppeteer']
  ```
  
  这种策略的问题是会将文本中所有出现过的独立片段都作为不同的 token，从而产生巨大的词表。而实际上词表中很多词是相关的，例如 “dog” 和 “dogs”、“run” 和 “running”，如果给它们赋不同的编号就无法表示出这种关联性。

  > 词表就是一个映射字典，负责将每个 token 映射到对应的编号 (IDs)，编号从 0 开始，一直到词表中所有 token 的数量，神经网络模型就是通过这些 token IDs 来区分每一个 token。

  当遇到不在词表中的词时，分词器会使用一个专门的 `[UNK]` token 来表示它是 unknown 的。显然，如果分词结果中包含很多 `[UNK]` token 就意味着丢掉了很多文本信息，因此**一个好的分词策略，应该尽可能不出现 unknown tokens**。

- **按字符切分 (Character-based)** 按更细的粒度进行分词，比如按字符切分。

  <img src="/assets/img/transformers-note-2/character_based_tokenization.png" alt="character_based_tokenization" style="display: block; margin: auto; width: 600px">

  这种策略把文本切分为字符而不是词语，这样就只会产生一个非常小的词表，并且很少会出现词表外的 tokens。

  但是从直觉上来看，字符本身并没有太大的意义，因此将文本切分为字符之后就会变得不容易理解。这也与语言有关，例如中文字符会比拉丁字符包含更多的信息，相对影响较小。此外，这种方式切分出的 tokens 会很多，例如一个由 10 个字符组成的单词就会输出 10 个 tokens，而实际上它们只是一个词。

  因此现在广泛采用的是一种同时结合了按词切分和按字符切分的方式——按子词切分 (Subword tokenization)。

- **按子词 (Subword) 切分** 高频词直接保留，低频词被切分为更有意义的子词。

- 例如 “annoyingly” 就是一个低频词，可以切分为 “annoying” 和 “ly”，这两个子词不仅出现频率更高，而且词义也得以保留。下图就是对文本 “Let’s do tokenization!“ 按子词切分的例子：

  <img src="/assets/img/transformers-note-2/bpe_subword.png" alt="bpe_subword" style="display: block; margin: auto; width: 600px">

  可以看到，“tokenization” 被切分为了 “token” 和 “ization”，不仅保留了语义，而且只用两个 token 就表示了一个长词。这中策略只用一个较小的词表就可以覆盖绝大部分的文本，基本不会产生 unknown tokens。尤其对于土耳其语等黏着语言，可以通过串联多个子词构成几乎任意长度的复杂长词。

### 加载与保存分词器

分词器的加载与保存与模型非常相似，也是使用 `from_pretrained()` 和 `save_pretrained()` 函数。例如，使用 `BertTokenizer` 类加载并保存 BERT 模型的分词器：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
```

与 `AutoModel` 类似，在大部分情况下，我们都应该使用 `AutoTokenizer` 类来加载分词器，它会根据 checkpoint 来自动选择对应的分词器：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.save_pretrained("./models/bert-base-cased/")
```

调用 `Tokenizer.save_pretrained()` 函数会在保存路径下创建三个文件：

- special_tokens_map.json：配置文件，里面包含 unknown tokens 等特殊字符的映射关系；
- tokenizer_config.json：配置文件，里面包含构建分词器需要的参数；
- vocab.txt：词表，每一个 token 占一行，行号就是对应的 token ID（从 0 开始）。

### 编码与解码文本

完整的文本编码 (Encoding) 过程实际上包含两个步骤：

1. **分词：**使用分词器按某种策略将文本切分为 tokens；
2. **映射：**将 tokens 转化为对应的 token IDs。 

> 因为不同预训练模型采用的分词策略并不相同，因此我们需要通过向 `Tokenizer.from_pretrained()` 函数传递模型 checkpoint 的名称来加载对应的分词器和词表。

下面，我们尝试使用 BERT 分词器来对文本进行分词：

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

可以看到，BERT 分词器采用的是子词 (subword) 切分策略：它会不断切分词语直到获得词表中的 token，例如 “transformer” 会被切分为 “transform” 和 “##er”。

然后，我们通过 `convert_tokens_to_ids()` 将切分出的 tokens 转换为对应的 token IDs：

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```
[7993, 170, 13809, 23763, 2443, 1110, 3014]
```

还可以通过 `encode()` 函数将这两个步骤合并，并且 `encode()` 会自动添加模型需要的特殊字符。例如对于 BERT 会自动在 token 序列的首尾分别添加 `[CLS]` 和 `[SEP]` token：

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

其中 101 和 102 分别是 `[CLS]` 和 `[SEP]`  对应的 token IDs。

**注意：**实际编码文本时，更为常见的是直接使用分词器进行处理。这样返回的结果中不仅包含处理后的 token IDs，还包含模型需要的其他辅助输入。例如对于 BERT 模型还会自动在输入中添加 `token_type_ids` 和 `attention_mask`：

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

文本解码 (Decoding) 与编码相反，负责将 token IDs 转化为原来的字符串。注意，解码过程不是简单地将 token IDs 映射回 tokens，还需要合并那些被分词器分为多个 token 的单词。下面我们尝试通过 `decode()` 函数解码前面生成的 token IDs：

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

解码文本是一个重要的步骤，当我们运用模型来预测新的文本时，都会调用这一函数。例如根据模板 (prompt) 生成文本、翻译或者摘要等 seq2seq 问题等等。

## 5.3 处理多段文本

在实际应用中，我们往往需要同时处理大量长度各异的文本。而且所有的神经网络模型都只接受批 (batch) 数据作为输入，即使只输入一段文本，也需要先将它组成只包含一个样本的 batch，然后才能送入模型，例如：

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

这里我们通过 `[ids]` 手工为输入增加了一个 batch 维（这个 batch 只包含一段文本），更多情况下送入的是包含多段文本的 batch：

```python
batched_ids = [ids, ids, ids, ...]
```

**再次强调：**上面的演示只是为了便于我们更好地理解分词背后的原理。实际应用中，我们应该直接使用分词器对文本进行处理，例如对于上面的例子：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print("Input IDs:\n", tokenized_inputs["input_ids"])

output = model(**tokenized_inputs)
print("Logits:\n", output.logits)
```

```
Input IDs:
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
Logits:
tensor([[-1.5607,  1.6123]], grad_fn=<AddmmBackward0>)
```

可以看到，分词器输出的结果字典中，token IDs 只是其中的一项（`input_ids`），字典中还会包含其他的输入项。前面我们之所以只输入 token IDs 模型也能正常运行，是因为它自动地补全了其他的输入项，例如 `attention_mask` 等，后面我们会具体介绍。

> 小提示：下面的例子中，分词器自动在 token 序列的首尾添加了 `[CLS]` 和 `[SEP]` token，所以上面两个例子中模型的输出是有差异的。因为 DistilBERT 在预训练时的输入中就包含 `[CLS]` 和 `[SEP]`，所以下面例子才是正确的使用方法。

### Padding 操作

将多段文本按批 (batch) 输入会产生的一个直接问题就是：batch 中的文本有长有短，而输入张量 (tensor) 必须是严格的二维矩形，维度为 `(batch size, token IDs sequence length)`，换句话说每一个文本编码后的 token IDs 的数量必须一样多。例如下面的 ID 列表是无法转换为张量的：

```python
batched_ids = [
    [200, 200, 200],
    [200, 200]
]
```

我们需要通过 Padding 操作，在短序列的最后填充特殊的 *padding token*，使得 batch 中所有的序列都具有相同的长度，例如：

```python
padding_id = 100

batched_ids = [
    [200, 200, 200],
    [200, 200, padding_id],
]
```

每个预训练模型使用的 padding token 的 ID 可能有所不同，可以通过其对应分词器的 `pad_token_id` 属性获得。下面我们尝试将两段文本分别以独立以及组成 batch 的形式送入到模型中：

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

`问题出现了！`{:.error} 在组成 batch 后，使用 padding token 填充的序列的结果出现了问题，与单独送入模型时的预测结果不同。这是因为 Transformer 模型会编码输入序列中的每一个 token 以建模完整的上下文，因此这里会将填充的 padding token 也当成是普通 token 一起编码，从而生成了不同的上下文语义表示。

因此，在进行 Padding 操作的同时，我们必须明确地告诉模型哪些 token 是我们填充的，它们不应该参与编码，这就需要使用到 attention mask。

> 在前面例子中，除了 token IDs 之外，我们还经常能看到一个 `attention_mask` 项，这就是下面要介绍的 Attention Mask。

### Attention masks

Attention masks 是一个与 input IDs 尺寸完全相同的仅由 0 和 1 组成的张量，其中 0 表示对应位置的 token 是填充符，不应该参与 attention 层的计算，而应该只基于 1 对应位置的 token 来建模上下文。

> 除了标记填充字符位置以外，许多特定的模型结构也会使用 Attention masks 来遮蔽掉一些 tokens。

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

**再次提醒：**这里只是为了演示。实际使用时，应该直接使用分词器对文本进行处理，它不仅会向 token 序列中添加 `[CLS]`、`[SEP]` 等特殊字符，还会自动地生成对应的 Attention masks。

目前大部分 Transformer 模型只能处理长度为 512 或 1024 的 token 序列，如果你需要处理的序列长度大于 1024，有以下两种处理方法：

- 使用一个支持长文的 Transformer 模型，例如 [Longformer](https://huggingface.co/transformers/model_doc/longformer.html)和  [LED](https://huggingface.co/transformers/model_doc/led.html)（最大长度 4096）；

- 设定一个最大长度 `max_sequence_length` 以**截断**输入序列：`sequence = sequence[:max_sequence_length]`。


### 直接使用分词器

前面我们介绍了分词、转换 token IDs、padding、构建 attention masks 以及截断等操作。实际上，直接使用分词器就能实现所有的这些操作。

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

可以看到，分词器的输出包含了模型需要的所有输入项。例如对于 DistilBERT 模型，就是 input IDs（`input_ids`）和 Attention mask（`attention_mask`）。

**padding 操作**通过 `padding` 参数来控制：

- `padding="longest"`： 将 batch 内的序列填充到当前 batch 中最长序列的长度；
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

可以看到在 `padding=True, truncation=True` 这样的设置下，同一个 batch 中的序列都会 pad 到相同的长度，并且大于模型最大接受长度的序列会被自动截断。

### 编码句子对

在上面的例子中，我们都是对单个序列进行编码（即使通过 batch 处理多段文本，也是并行地编码单个序列），而实际上对于 BERT 等包含“句子对”分类预训练任务的模型来说，都支持对“句子对”进行编码，例如：

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

可以看到分词器自动使用 `[SEP]` token 拼接了两个句子，输出形式为“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]} \text{ sentence2 } \texttt{[SEP]}$”的 token 序列，这也是 BERT 模型预期的输入格式。返回结果中除了前面我们介绍过的 `input_ids` 和 `attention_mask` 之外，还包含了一个 `token_type_ids` 项，用于标记输入序列中哪些 token 属于第一个句子，哪些属于第二个句子。对于上面的例子，如果我们将 `token_type_ids` 项与 token 序列对齐：

```
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
[      0,      0,    0,     0,       0,          0,   0,       0,      1,    1,     1,        1,     1,   1,       1]
```

可以看到第一个句子“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]}$”片段所有 tokens 的 token type ID 都为 0，而第二个句子“$\text{sentence2 } \texttt{[SEP]}$”片段对应的 token type ID 都是 1。

> 如果我们选择其他的预训练模型，分词器的输出不一定会包含 `token_type_ids` 项（例如 DistilBERT 模型）。分词器的输出格式只需保证与模型在预训练时的输入格式保持一致即可。

实际使用时，我们不需要去关注编码结果中是否包含  `token_type_ids` 项，分词器会根据 checkpoint 自动调整适用于对应模型的格式，例如：

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence1_list = ["This is the first sentence 1.", "second sentence 1."]
sentence2_list = ["This is the first sentence 2.", "second sentence 2."]

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
    [ 101, 2023, 2003, 1996, 2034, 6251, 1015, 1012,  102, 2023, 2003, 1996,
     2034, 6251, 1016, 1012,  102],
    [ 101, 2117, 6251, 1015, 1012,  102, 2117, 6251, 1016, 1012,  102,    0,
        0,    0,    0,    0,    0]]), 
 'token_type_ids': tensor([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}

torch.Size([2, 17])
```

可以看到分词器成功地输出了形式为“$\texttt{[CLS]} \text{ sentence1 } \texttt{[SEP]} \text{ sentence2 } \texttt{[SEP]}$”的 token 序列，并且将两个 token 序列都 pad 到了相同的长度。

## 参考

[[1]](https://huggingface.co/docs/transformers/index) Transformers 官方文档  
[[2]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程
