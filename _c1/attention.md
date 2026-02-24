---
title: 第三章：注意力机制
author: SHENG XU
date: 2022-05-04
category: nlp
layout: post
mathjax: yes
---

如前两章所述，Transformer 模型之所以如此强大，是因为它采用了一种特殊的结构——注意力机制（Attention）来建模文本，大幅加强了语义表征的上下文感知能力。尤其是对于大语言模型而言，正是注意力机制赋予了大语言模型理解上下文并生成连贯响应的能力。

那么注意力机制究竟有何魔力，它具体是如何工作的？本章将通过介绍常用的 Multi-Head Attention 为你揭晓答案，并且使用 Pytorch 框架手把手带你实现一个 Transformer block。

> **思考**
>
> 如果你不熟悉 Pytorch 可以跳过本章的代码部分，后续借助 Transformers 库可以方便地调用任何 Transformer 模型，而不必像本章一样手工编写。
{: .block-warning }

## 3.1 注意力机制

无论对于哪种 NLP 模型，理解用户输入始终是开展工作的第一步，如之前在 2.5.2 节中介绍的那样，常规的做法就是首先对输入文本进行分词，然后将每个词元（Token）都转化为对应的词嵌入（Embeddings），这样文本就转换为一个由词嵌入组成的矩阵 $\boldsymbol{X}=(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n)$，其中 $\boldsymbol{x}_i$ 就表示第 $i$ 个词元的嵌入，维度为 $d$，故 $\boldsymbol{X}\in \mathbb{R}^{n\times d}$。

在 Transformer 模型提出之前，对序列 $\boldsymbol{X}$ 的常规编码方式是通过循环神经网络（RNNs）或卷积神经网络（CNNs）来进行。


- 循环神经网络（例如 LSTM）的方案很简单，每一个词 $\boldsymbol{x}_t$ 对应的编码结果 $\boldsymbol{y}_t$ 都基于这个词前面一个词的表示获得，通过递归地计算得到（也就是基于词的上文信息来计算词嵌入）：
  
  $$
  \boldsymbol{y}_t =f(\boldsymbol{y}_{t-1},\boldsymbol{x}_t)\tag{3.1}
  $$
  
  循环神经网络的序列建模方式虽然与人类阅读类似，但是递归的结构导致其无法并行计算，在处理长序列时还可能会出现梯度爆炸或梯度消失问题，而且循环神经网络本质是一个马尔科夫决策过程，难以学习到全局的结构信息；

- 卷积神经网络则运用滑动窗口基于局部上下文来编码文本，例如核尺寸为 3 的卷积操作就是使用每一个词自身以及前一个和后一个词来生成该词的语义嵌入：
  
  $$
  \boldsymbol{y}_t = f(\boldsymbol{x}_{t-1},\boldsymbol{x}_t,\boldsymbol{x}_{t+1})\tag{3.2}
  $$
  
  卷积神经网络能够并行计算，因此速度很快，但是由于是通过窗口来进行编码，所以更侧重于捕获局部信息（窗口内词元之间的交互），难以建模远距离词之间的依赖。

直到 2017 年 Google 在[《Attention is All You Need》](https://arxiv.org/abs/1706.03762)中提供了第三个方案：直接使用注意力机制编码整个文本。相比循环神经网络要逐步递归才能获得全局信息（因此一般使用双向 RNN），而卷积神经网络只能获取局部信息，需要通过堆叠多层来增大感受野，注意力机制一步到位获取了全局信息：

$$
\boldsymbol{y}_t = f(\boldsymbol{x}_t,\boldsymbol{A},\boldsymbol{B})\tag{3.3}
$$

其中 $\boldsymbol{A},\boldsymbol{B}$ 是另外的词嵌入序列（矩阵），如果取 $\boldsymbol{A}=\boldsymbol{B}=\boldsymbol{X}$ 就称为自注意力（Self-Attention），即直接将 $\boldsymbol{x}_t$ 与自身序列中的每个词进行比较，最后算出 $\boldsymbol{y}_t$。

### 3.1.1 缩放点积注意力

基于注意力思路编码文本并非由 Google 首创，在早期的神经网络工作中许多研究者就已经提出了各种版本的注意力机制。直到 2017 年 Transformer 模型横空出世，才确定了注意力机制的形式，并被后续的工作所沿用，这种注意力机制的全称为缩放点积注意力（Scaled Dot-product Attention），如图 3-1 所示。

<img src="/assets/img/attention/attention.png" style="display: block; margin: auto; width: 250px">

<center>图 3-1 缩放点积注意力机制</center>

注意力机制的工作过程主要包含两个步骤：

1. **计算注意力权重**：使用某种相似度函数度量每一个查询（query）向量和所有键（key）向量之间的关联程度。对于长度为 $m$ 的查询序列 Query 和长度为 $n$ 的键序列 Key，该步骤会生成一个尺寸为 $m \times n$ 的注意力分数矩阵。特别地，注意力机制使用点积作为相似度函数，这样相似的查询和键就会具有较大的点积。

   由于点积可以产生任意大的数字，这会破坏训练过程的稳定性。因此注意力分数还需要乘以一个缩放因子来标准化它们的方差，然后用一个 softmax 函数标准化。这样就得到了最终的注意力权重 $w_{ij}$，表示第 $i$ 个查询向量与第 $j$ 个键向量之间的关联程度。

2. **更新词嵌入：**将权重 $w\_{ij}$ 与对应的值（value）向量 $\boldsymbol{v}\_1,...,\boldsymbol{v}\_n$ 相乘以获得第 $i$ 个查询向量更新后的语义表示 $\boldsymbol{x}\_i' = \sum_{j} w\_{ij}\boldsymbol{v}\_j$。

形式化表示为：

$$
\text{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V} \tag{3.4}
$$

其中 $\boldsymbol{Q}\in\mathbb{R}^{m\times d_k}, \boldsymbol{K}\in\mathbb{R}^{n\times d_k}, \boldsymbol{V}\in\mathbb{R}^{n\times d_v}$ 分别是查询（query）、键（key）、值（value）向量序列。如果忽略 softmax 激活函数，实际上它就是三个 $m\times d_k,d_k\times n, n\times d_v$ 矩阵相乘，得到一个 $m\times d_v$ 的矩阵，也就是将 $m\times d_k$ 的序列 $\boldsymbol{Q}$ 编码成了一个新的 $m\times d_v$ 的序列。

将上面的公式拆开来看更加清楚：

$$
\text{Attention}(\boldsymbol{q}_t,\boldsymbol{K},\boldsymbol{V}) = \sum_{s=1}^n \frac{1}{Z}\exp\left(\frac{\langle\boldsymbol{q}_t, \boldsymbol{k}_s\rangle}{\sqrt{d_k}}\right)\boldsymbol{v}_s \tag{3.5}
$$

其中 $Z$ 是归一化因子，$\boldsymbol{K},\boldsymbol{V}$ 是一一对应的键和值向量序列，注意力机制就是通过 $\boldsymbol{q}_t$ 这个查询与各个键 $\boldsymbol{k}_s$ 内积并 softmax 的方式来得到 $\boldsymbol{q}_t$ 与各个值 $\boldsymbol{v}_s$ 的相似度，然后加权求和，得到一个 $d_v$ 维的向量。其中因子 $\sqrt{d_k}$ 起到调节作用，使得内积不至于太大。

下面我们通过 Pytorch 来手工实现注意力机制。首先需要将文本分词为词元序列，然后将每一个词元转换为对应的词嵌入。Pytorch 提供了 `torch.nn.Embedding` 层来完成该操作，即构建一个从词元 ID 到词元嵌入的映射表：

```python
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())
```

```
tensor([[ 2051, 10029,  2066,  2019,  8612]])
Embedding(30522, 768)
torch.Size([1, 5, 768])
```

这里为了演示方便，通过设置 `add_special_tokens=False` 去除了分词结果中的 `[CLS]` 和 `[SEP]`。可以看到，BERT 模型对应的词表大小为 30522，每个词语的词向量维度为 768。嵌入（Embedding）层把输入的词语序列映射到了尺寸为 `[batch_size, seq_len, hidden_dim]` 的张量。

接下来就是创建查询、键、值向量序列 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$，并且使用点积作为相似度函数来计算注意力分数：

```python
import torch
from math import sqrt

Q = K = V = inputs_embeds
dim_k = K.size(-1)
scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
print(scores.size())
```

```
torch.Size([1, 5, 5])
```

这里 $\boldsymbol{Q},\boldsymbol{K}$ 的序列长度都为 5，因此生成了一个 $5\times 5$ 的注意力分数矩阵，接下来就是应用 softmax 函数标准化注意力权重：

```python
import torch.nn.functional as F

weights = F.softmax(scores, dim=-1)
print(weights)
print(weights.sum(dim=-1))
```

```
tensor([[[1.0000e+00, 5.9045e-13, 1.4973e-13, 3.8262e-14, 8.8257e-14],
         [1.9624e-12, 1.0000e+00, 5.1714e-13, 2.5073e-13, 3.2403e-13],
         [7.5319e-14, 7.8275e-14, 1.0000e+00, 4.0099e-14, 8.0383e-14],
         [1.4938e-12, 2.9453e-12, 3.1119e-12, 1.0000e+00, 9.5267e-12],
         [5.1650e-12, 5.7057e-12, 9.3512e-12, 1.4281e-11, 1.0000e+00]]],
       grad_fn=<SoftmaxBackward0>)
tensor([[1., 1., 1., 1., 1.]], grad_fn=<SumBackward1>)
```

可以看到，由于这里查询、键、值向量序列都为输入序列本身，因此对角线计算出的注意力分数都非常大，因为每个词嵌入都与自身完全一致（点积为 1），并且每一个查询的注意力分数求和为 1，验证了通过 softmax 函数成功进行了标准化。

最后将注意力权重与值序列相乘，就实现了一个简化版的注意力机制：

```python
attn_outputs = torch.bmm(weights, V)
print(attn_outputs.shape)
```

```
torch.Size([1, 5, 768])
```

可以将上面这些操作封装为函数以方便后续调用：

```python
import torch
import torch.nn.functional as F
from math import sqrt

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
```

注意，上面的代码还考虑了 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 序列的掩码（Mask）。正如之前在 2.3.3 节中介绍的那样，注意力机制还会使用注意力掩码（Attention Mask）来遮盖掉某些特殊词以防止模型关注它们，例如特殊填充词（padding）就不应该参与计算，因此会将这些特殊词对应的注意力分数设置为 $-\infty$，这样 softmax 之后其对应的注意力权重就为 0 了（$e^{-\infty}=0$）。

但是，从上面的例子中就可以发现：当 $\boldsymbol{Q}$ 和 $\boldsymbol{K}$ 序列相同时，注意力机制会为上下文中相同的词分配非常大的分数（点积为 1），而在实践中，相关词往往比相同词更重要。例如对于上面的例子，只有关注“time”和“arrow”才能够确认“flies”的含义。为了解决这一问题，研究者又提出了多头注意力机制。

### 3.1.2 多头注意力

多头注意力（Multi-head Attention）机制首先通过线性映射将 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 序列映射到特征空间，每一组线性投影后的向量表示称为一个头（head），然后在每组映射后的序列上再应用上一节中介绍的缩放点积注意力，如图 3-2 所示：

<img src="/assets/img/attention/multi_head_attention.png" style="display: block; margin: auto; width: 250px">

<center>图 3-2 多头注意力机制</center>

每个注意力头负责关注某一方面的语义相似性，多个头就可以让模型同时关注多个方面。因此与简单的缩放点积注意力机制相比，多头注意力机制可以捕获到更加复杂的特征信息。

形式化表示为：

$$
\begin{gather}head_i = \text{Attention}(\boldsymbol{Q}\boldsymbol{W}_i^Q,\boldsymbol{K}\boldsymbol{W}_i^K,\boldsymbol{V}\boldsymbol{W}_i^V)\\\text{MultiHead}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = \text{Concat}(head_1,...,head_h)\end{gather} \tag{3.6}
$$

其中 $\boldsymbol{W}_i^Q\in\mathbb{R}^{d_k\times \tilde{d}_k}, \boldsymbol{W}_i^K\in\mathbb{R}^{d_k\times \tilde{d}_k}, \boldsymbol{W}_i^V\in\mathbb{R}^{d_v\times \tilde{d}_v}$ 是映射矩阵，$h$ 是注意力头的数量。最后，将多头的结果拼接起来就得到最终 $m\times h\tilde{d}_v$ 的结果序列。所谓的“多头”（Multi-head），其实就是多执行几次注意力机制，然后把结果拼接。

下面我们首先实现一个注意力头：

```python
from torch import nn

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs
```

每个头都会初始化三个独立的线性层，负责将 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 序列映射到尺寸为  `[batch_size, seq_len, head_dim]` 的张量，其中 `head_dim` 是映射到的向量维度。实践中一般将 `head_dim` 设置为 `embed_dim` 的因数，这样 token 嵌入式表示的维度就可以保持不变，例如 BERT 有 12 个注意力头，就可以把每个头的维度被设置为 $768 / 12 = 64$。

最后只需要简单地拼接多个注意力头的输出就可以构建出多头注意力层了（这里在拼接后还通过一个线性变换来生成最终的输出张量）：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x
```

下面我们使用 BERT 模型的参数初始化构建出的多头注意力层，并且将之前构建好的输入送入模型以验证是否工作正常：

```python
from transformers import AutoConfig
from transformers import AutoTokenizer

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs_embeds = token_emb(inputs.input_ids)

multihead_attn = MultiHeadAttention(config)
query = key = value = inputs_embeds
attn_output = multihead_attn(query, key, value)
print(attn_output.size())
```

```
torch.Size([1, 5, 768])
```

可以看到模型按照我们的预期输出了维度为 $5\times 768$ 的编码结果。

## 3.2 Transformer 编码器

回忆一下上一章中介绍过的标准 Transformer 模型，左侧的编码器（Encoder）负责将输入的词序列转换为词嵌入序列，以获得上下文感知的输入语义信息，右侧的解码器（Decoder）则基于编码器输出的语义表示（隐状态）以及其他输入来迭代地生成词序列作为输出（每次生成一个词），如图 3-3 所示。

<img src="/assets/img/attention/transformer.jpeg" alt="transformer" style="display: block; margin: auto; width: 700px">

<center>图 3-3 标准 Transformer 模型</center>

其中，编码器和解码器都各自包含有多层构件（building blocks）。图 3-4 展示了一个翻译任务的例子：

<img src="/assets/img/attention/encoder_decoder_architecture.png" alt="encoder_decoder_architecture" style="display: block; margin: auto; width: 700px">

<center>图 3-4 Transformer 模型执行翻译任务</center>

可以看到：

- 输入的词首先被转换为词嵌入。由于注意力机制无法捕获词之间的位置关系，因此还通过位置编码（Positional Embeddings，PE）向输入中添加位置信息；
- 编码器由一堆编码器编码构件（encoder layers）组成，类似于图像领域中的堆叠卷积层。同样地，在解码器中也包含有堆叠的解码构件（decoder layers）；
- 编码器的输出被送入到解码器层中以预测概率最大的下一个词，然后当前的词序列又被送回到解码器中以继续生成下一个词，重复直至出现序列结束符 EOS 或者超过最大输出长度。

### 3.2.1 前馈网络层

Transformer 编码器/解码器中的前馈网络子层（Feed-Forward Network, FFN）实际上就是两层全连接神经网络，负责对嵌入序列进行非线性变换。具体来说，对于给定的输入 $\boldsymbol{X}$，前馈网络层由两个线性变换和一个非线性激活函数组成：
$$
FFN(\boldsymbol{X}) = \sigma(\boldsymbol{X}\boldsymbol{W}^U + \boldsymbol{b}_1)\boldsymbol{W}^D + \boldsymbol{b}_2\tag{3.7}
$$
其中 $\boldsymbol{W}^U\in \mathbb{R}^{d\times d'}$ 和 $\boldsymbol{W}^D\in \mathbb{R}^{d'\times d}$ 分别是第一层和第二层的线性变换权重矩阵，$\boldsymbol{b}_1\in\mathbb{R}^{d'}$ 和 $\boldsymbol{b}_2\in\mathbb{R}^{d}$ 是偏置项，$\sigma$ 是激活函数（在原始 Transformer 中采用 ReLU 作为激活函数）。常见做法是让第一层的维度是词向量大小的 4 倍，然后以 GELU 作为激活函数。

下面实现一个简单的前馈网络子层（这里省略了偏置项）：

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
```

将前面注意力层的输出送入到该层中以测试是否符合我们的预期：

```python
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
print(ff_outputs.size())
```

```
torch.Size([1, 5, 768])
```

至此创建完整 Transformer 编码器的所有要素都已齐备，只需要再加上跳跃连接（Skip Connections）和层归一化（Layer Normalization）就大功告成了。

> **延伸**
>
> 跳跃连接（Skip Connections）又称残差连接（Residual Connection），是深度神经网络中一种至关重要的架构设计，其核心思想是‌在网络的某些层之间创建“快捷路径”，允许输入信息直接跳过一个或多个中间层，传递到更深的层。
>
> 这种设计打破了传统神经网络中信息必须逐层顺序传递的模式，有助于‌缓解梯度消失/爆炸问题、防止网络退化以及避免早期信息的丢失‌。
{: .block-tip }

### 3.2.2 层归一化

层归一化（Layer Normalization）负责将一个批次（batch）输入中的每一个都标准化为均值为零且具有单位方差。跳跃连接则是将张量直接传递给模型的下一层而不进行处理，并将其添加到处理后的张量中。

目前有层后归一化和层前归一化两种常见的向 Transformer 编码器/解码器中添加归一化的方式，如图 3-5 所示：

- **层后归一化（Post layer normalization，Post-Norm）**：Transformer 论文中使用的方式，将层归一化放在跳跃连接之间。 但是因为梯度可能会发散，这种做法很难训练，还需要结合学习率预热（learning rate warm-up）等技巧；
- **层前归一化（Pre layer normalization, Pre-Norm）**：目前主流的做法，将层归一化放置于跳跃连接的范围内。这种做法通常训练过程会更加稳定，并且不需要任何学习率预热。

<img src="/assets/img/attention/arrangements_of_layer_normalization.png" alt="arrangements_of_layer_normalization" style="display: block; margin: auto; width: 600px">

<center>图 3-5 两种添加层归一化的方式</center>

这里采用第二种方式来构建 Transformer 编码层：

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
```

同样地，这里将之前构建的输入送入到该层中进行测试：

```python
encoder_layer = TransformerEncoderLayer(config)
print(inputs_embeds.shape)
print(encoder_layer(inputs_embeds).size())
```

```
torch.Size([1, 5, 768])
torch.Size([1, 5, 768])
```

结果符合预期！至此，我们就构建出了一个几乎完整的 Transformer 编码层。

### 3.2.3 位置编码

如前面所述，由于注意力机制无法捕获词之间的位置信息，因此 Transformer 模型还使用位置编码（Positional Embeddings，PE）来添加词的位置信息。位置编码基于一个简单但有效的想法：使用与位置相关的值模式来增强词嵌入。

如果预训练数据集足够大，那么最简单的方法就是让模型自动学习位置嵌入。下面本章就以这种方式创建一个自定义的嵌入（Embeddings）模块，它同时将词和位置映射到嵌入表示，最终的输出是两个表示之和：

```python
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

embedding_layer = Embeddings(config)
print(embedding_layer(inputs.input_ids).size())
```

```
torch.Size([1, 5, 768])
```

除此以外，位置编码还有一些替代方案：

**绝对位置编码**：使用由调制的正弦和余弦信号组成的静态模式来编码位置。 当没有大量训练数据可用时，这种方法尤其有效；

**相对位置编码**：在生成某个词的词嵌入时，一般距离它近的词更为重要，因此也有工作采用相对位置编码。因为每个词的相对嵌入会根据序列的位置而变化（根据键和查询之间的偏移量进行计算），这需要在模型层面对注意力机制进行修改，而不是通过引入嵌入层来完成，例如 DeBERTa 等模型。

**旋转位置编码（Rotary Position Embedding，RoPE）**：RoPE 使用了基于绝对位置信息的旋转矩阵来表示注意力中的相对位置信息，为序列中每个绝对位置设置了特定的旋转矩阵，并和该位置的查询和键进行相乘。由于 RoPE 性能较好且具有长期衰减特性，目前被大语言模型广泛采用。

**ALiBi 位置编码**：ALiBi 是一种特殊的相对位置编码，主要用于增强 Transformer 模型的长度外推能力。具体来说，在原始注意力计算公式长，ALiBi 进一步引入了与相对距离成比例关系的惩罚因子来调整注意力分数。ALiBi 具有优秀的长度外推能力，能够对超过上下文窗口长度的文本进行有效建模。

下面我们将所有这些层结合起来构建完整的 Transformer 编码器：

```python
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
```

同样地，我们对该层进行简单的测试：

```python
encoder = TransformerEncoder(config)
print(encoder(inputs.input_ids).size())
```

```
torch.Size([1, 5, 768])
```

## 3.3 Transformer 解码器

Transformer 模型中的解码器基于编码器最后一层输出以及已生成的词元序列进行后续内容的生成。因此，与编码器只有一种注意力子层（多头注意力机制）不同，Transformer 模型的解码器包含两种注意力子层，如图 3-6 所示：

<img src="/assets/img/attention/transformer_decoder.png" alt="transformer_decoder" style="display: block; margin: auto; width: 600px">

<center>图 3-6 Transformer 解码器结构</center>

- **掩码多头注意力层（Masked multi-head self-attention layer）**：在计算注意力时引入掩码操作，确保在每个时间步生成的词仅基于过去的输出和当前的输入（不依赖未来信息），否则解码器就相当于作弊了；
- **交叉注意力层（Cross attention layer）**：以解码器的中间表示作为查询，对编码器的输出 键和值向量执行多头注意力计算。通过这种方式，交叉注意力层就可以学习到如何关联来自两个不同序列的词，例如两种不同的语言。需要说明的是，解码器可以访问每个 block 中 编码器的键和值。

与编码器中的 Mask 不同，解码器的 Mask 是一个下三角矩阵：

```python
seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
print(mask[0])
```

```
tensor([[1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.]])
```

这里使用 PyTorch 自带的 `tril()` 函数来创建下三角矩阵，然后同样地，通过 `Tensor.masked_fill()` 将所有零替换为负无穷大来防止注意力头看到未来的词而造成信息泄露：

```python
scores.masked_fill(mask == 0, -float("inf"))
```

```
tensor([[[26.8082,    -inf,    -inf,    -inf,    -inf],
         [-0.6981, 26.9043,    -inf,    -inf,    -inf],
         [-2.3190,  1.2928, 27.8710,    -inf,    -inf],
         [-0.5897,  0.3497, -0.3807, 27.5488,    -inf],
         [ 0.5275,  2.0493, -0.4869,  1.6100, 29.0893]]],
       grad_fn=<MaskedFillBackward0>)
```

这里我们对解码器只做简单的介绍，如果你想更深入了解，可以参考 Andrej Karpathy 实现的 [minGPT](https://github.com/karpathy/minGPT)。

## 3.4 注意力机制变体

大多数 Transformer 模型都使用我们上面介绍的标准注意力机制，通过成对的方式进行序列数据的语义建模（考虑所有词元之间的交互），即注意力矩阵是方阵，因此对于长度为 $n$ 的序列，注意力机制的计算复杂度为  $O(n^2)$。当处理长文本时，这会成为严重的计算瓶颈。

为此，研究者又提出了一些注意力机制的变体以降低计算复杂度，提高效率。下面介绍几种较为常见的注意力机制变体。

### 3.4.1 稀疏注意力机制

通常情况下，局部上下文（窗口内左侧和右侧的多个词元）足以对当前词元提供足够的信息，并且通过堆叠具有小窗口的注意力层，最后一层的感受野将不仅仅局限于窗口内的词元，而是能构建整个文本的表征。基于这种思路，纯编码器模型 Longformer 提出了一种局部注意力（local attention）机制：大部分词元仅基于窗口内的词元建模语义，而一些预先选定的特殊词元被赋予全局关注能够可以访问所有词元，并且这个过程是对称的，其注意力掩码如图 3-7 所示。

<img src="/assets/img/attention/local_attention_mask.png" alt="local_attention_mask" style="display: block; margin: auto; width: 300px">

<center>图 3-7 局部注意力机制</center>

这样只需要使用参数较少的注意力矩阵，就可以让模型处理序列长度更大的输入。后来这种做法被大语言模型广泛采用，称为滑动窗口注意力机制（sliding window attention），它会设置一个长度为 $w$ 的窗口，对于每个词元 $x_t$，通过掩码机制只需要关注位于它之前窗口内的词元 $[x_{t-w+1},...,x_t]$，从而将复杂度降低到 $O(wn)$。并且通过信息的逐层传递，模型具有随层数线性增长的感受野，从而实现了远处词元信息的获取，如图 3-8 所示。

<img src="/assets/img/attention/sliding_window_attention.png" alt="sliding_window_attention" style="display: block; margin: auto; width: 600px">

<center>图 3-8 滑动窗口注意力机制</center>

### 3.4.2 多查询注意力

还有一些研究者从改进注意力机制本身出发，提出了[多查询注意力](https://arxiv.org/abs/1911.02150)（Multi-Query Attention，MQA），针对不同的头共享相同的键和值变换矩阵，这种方法通过减少访存量来实现计算速度的提升，对模型性能产生的影响也较小。

后来，为了结合多查询注意力机制的效率和多头注意力机制的性能，研究者进一步提出了分组查询注意力机制（Grouped-Query Attention，GQA）。分组查询注意力机制将全部的头划分为若干组，同一组内的头共享相同的变换矩阵，从而有效平衡了效率和性能。图 3-9 展示了这两种注意力机制与标准多头注意力机制的区别。

<img src="/assets/img/attention/mha_mqa_gqa.png" alt="sliding_window_attention" style="display: block; margin: auto; width: 400px">

<center>图 3-9 多种注意力机制的对比</center>

### 3.4.3 硬件优化注意力机制

除了在算法层面上进行改进，还有一些研究者尝试在系统层面来优化注意力模块的计算效率和显存消耗，其中两个具有代表性的工作是 [FlashAttention](https://arxiv.org/abs/2205.14135) 与 [PagedAttention](https://arxiv.org/abs/2309.06180)。其中 FlashAttention 通过矩阵分块计算以及减少内存读写次数的方式，提高了注意力分数的计算效率。而 PagedAttention 则针对解码阶段，对于键值缓存（key-value cache）进行分块存储并优化了计算方式从而提高了计算效率。

FlashAttention 的核心思想是尽可能减少对中间结果的保存，直接得到最终结果。如前面所述，注意力机制的计算方法为 $\text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}$，其中需要保留多个中间结果，如 $\boldsymbol{Q}\boldsymbol{K}^{\top}$ 和 softmax 后的注意力分布矩阵。这些中间结果需要频繁写入显存导致大量的显存读写操作。FlashAttention 通过矩阵分块和算子融合等方法，将中间结果一直保存在缓存中，直到获得最终结果后再写回显存中，从而减少了显存读写量。

PagedAttention 则是针对键值缓存拼接和注意力计算进行了优化，显著降低了这两部分运算的访存量，从而提高计算效率。

- 传统的键值缓存拼接操作会在每次拼接时新分配一段显存空间，然后将原始键值缓存和新生成词元的隐状态复制到新分配的显存中，导致反复读写显存，并产生较多的显存碎片。为此，PagedAttention 引入操作系统中显存分页管理的方法，预先将显存划分成若干块，给之后的键值缓存“预留空间”，显著减少了拼接时反复分配显存的操作。
- 在注意力计算方面，PagedAttention 通过提高计算的并行度来减少访存量。具体来说，增量解码阶段是以当前词作为查询向量，与已生分成序列进行注意力计算。PagedAttention 对前序生成文本的键值缓存采用分页管理操作，并使用算子融合方法将查询向量与多个分页的键值缓存并行计算，从而提升了计算效率。

## 3.5 新型模型架构

### 3.5.1 混合专家模型

除了改进注意力机制以外，还有一些研究者尝试引入基于稀疏激活的混合专家架构（Mixture-of-Experts，MoE）对前馈网络模块进行改进，通过将 Transformer 模块中的特定前馈层替换为 MoE 层，使得模型可以在推理过程中仅激活部分参数，从而在不显著提升计算成本的同时实现对模型参数的拓展。混合专家模型示意图如图 3-10 所示。

<img src="/assets/img/attention/mixture_of_experts.png" alt="mixture_of_experts" style="display: block; margin: auto; width: 600px">

<center>图 3-10 混合专家模型示意图</center>

在标准 Transformer 架构中，每个词元均通过单一的前馈神经网络进行处理，而 MoE 架构会在前馈模块内部部署多个独立的网络（而非单一网络），每个网络维护独立的权重参数，被称为"专家"。然后通过路由网络负责决策哪些词元分配给哪些专家处理。

具体来说，在 MoE 架构中，每个 MoE 层包含 $K$ 个专家组件，记为 $[E_1,E_2,...,E_K]$， 其中每个专家组件 $E_i$ 都是一个前馈神经网络。对于输入的每个词元表示 $\boldsymbol{x}_t$，模型首先通过一个路由网络（或称为门控函数）$G$ 来计算该词元对应于各个专家的权重：
$$
G(\boldsymbol{x}_t) = \text{softmax}(\text{topk}(\boldsymbol{x}_t \cdot \boldsymbol{W}^G))\tag{3.8}
$$
其中 $\boldsymbol{W}^G$ 是负责计算每个专家得分的线性映射，topk 表示只会选择概率最高的 $k$ 个专家进行激活，最后通过 softmax 函数计算出权重 $G(\boldsymbol{x}_t)=[G(\boldsymbol{x}_t)_1,...,G(\boldsymbol{x}_t)_k]$，没有被选择的专家权重将被置为 0。

最后，将被选择专家的输出加权和作为该 MoE 层的最终输出 $\boldsymbol{o}_t$：
$$
\boldsymbol{o}_t = \text{MoELayer}(\boldsymbol{x}_t) = \sum_{i=1}^k G(\boldsymbol{x}_t)_i\cdot E_i(\boldsymbol{x}_t) \tag{3.9}
$$

### 3.5.2 状态空间模型

此外，还有一些研究人员不满足于对 Transformer 模型进行改进，直接基于参数化状态空间模型（State Space Model，SSM）设计出了多种新型模型架构，在提高长文本建模效率的同时还保持了较好的序列建模能力。表 3-1 展示了一些代表性新型模型的复杂度，其中 $T$ 表示序列长度，$H$ 表示输入表示的维度，$N$ 表示状态空间模型压缩后的维度，$M$ 表示 Hyena 每个模块的层数。

<center>表 3-1 多个新型模型的复杂度对比</center>

| 模型        | 解码复杂度      | 训练复杂度             |
| ----------- | --------------- | ---------------------- |
| Transformer | $O(TH + H^2)$   | $O(T^2H + TH^2)$       |
| Mamba       | $O(N^2H + H^2)$ | $O(TN^2H + TH^2)$      |
| RWKV        | $O(H^2)$        | $O(TH^2)$              |
| RetNet      | $O(H^2)$        | $O(TH^2)$              |
| Hyena       | $O(TMH + MH^2)$ | $O(TMH\log T + TMH^2)$ |

参数化状态空间模型可以视为循环神经网络和卷积神经网络的结合体：它利用卷积计算对输入进行并行化编码，同时在计算中不需要访问前序的所有词元，只利用前一时刻的信息就可以自回归地进行预测。为了同时实现并行化计算和循环解码，状态空间模型在输入和输出之间引入了额外的状态变量。

> **延伸**
>
> 在实际应用中，为了更好地建模序列信息，基于状态空间模型的新型架构通常采用状态空间模型与前馈网络层交替堆叠的方式构建。
{: .block-tip }

尽管计算效率较高，但是状态空间模型的性能相比 Transformer 模型仍有差距。为此，一些研究者尝试改进状态空间模型以提高语言建模能力。一些代表性模型包括：

- **[Mamba](https://arxiv.org/abs/2312.00752)**：一种状态空间模型的变体，主要思想是在状态更新中引入基于当前输入的信息选择（Selection）机制，来确定当前时刻状态如何从前一时刻状态以及当前输入中提取信息。但是由于在状态计算过程中引入了非线性变换，Mamba 无法直接利用快速傅里叶变换实现高效卷积计算。
- **[RWKV](https://arxiv.org/abs/2305.13048)**：尝试结合 RNN 和 Transformer 的优点。其主要创新是在每层计算中使用词元偏移（Token Shift）来代替词元表示，并且将 Transformer 中的多头注意力模块和前馈网络模块分别替换为时间混合（Time-mixing）模块和频道混合 （Channel-mixing）模块。类似于 Mamba，RWKV 在解码过程中可以像 RNN 一样只参考前一时刻的状态。
- **[RetNet](https://arxiv.org/abs/2307.08621)**：提出多尺度保留（Multi-scale Retention, MSR）机制来代替多头注意力模块，在状态更新的线性映射中引入了输入相关信息来提升序列建模能力，RetNet 还可以通过类似注意力操作的矩阵乘法，对所有词元的状态进行并行化计算，因此同时保留了循环计算和并行计算的优点。
- **[Hyena](https://arxiv.org/abs/2302.10866)**：使用长卷积（Long Convolution）模块来替换注意力模块，并借助快速傅里叶变换提高卷积计算效率。

本章的所有代码已经整理于 Github：  
[https://gist.github.com/jsksxs360/3ae3b176352fa78a4fca39fff0ffe648](https://gist.github.com/jsksxs360/3ae3b176352fa78a4fca39fff0ffe648)

## 参考

[[1]](https://huggingface.co/course/chapter1/1) HuggingFace 在线教程  
[[2]](https://kexue.fm/archives/4765) 苏剑林. 2018.《Attention is All You Need》浅读（简介+代码）  
[[3]](https://github.com/nlp-with-transformers) Lewis Tunstall等. 2022. Natural Language Processing with Transformers. O’Reilly.  
[[4]](https://github.com/RUCAIBox/LLMSurvey) 赵鑫等. 2023. A Survey of Large Language Models
