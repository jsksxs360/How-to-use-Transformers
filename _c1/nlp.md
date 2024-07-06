---
title: 第一章：自然语言处理
author: SHENG XU
date: 2020-01-08
category: NLP
layout: post
mathjax: yes
---

自然语言处理（Natural Language Processing，NLP）是一门借助计算机技术研究人类语言的科学。虽然该领域的发展历史不长，但是其发展迅速并且取得了许多令人印象深刻的成果。

在上手实践之前，我想先给大家简单介绍一下自然语言处理的发展历史以及 Transformer 模型的概念，这对于后面理解模型结构会有很大帮助。本章将带大家快速穿越自然语言处理的发展史，了解从统计语言模型到大语言模型的发展历程。

## 1.1 自然语言处理发展史

自然语言处理的发展大致上可以分为两个阶段：

**第一阶段：不懂语法怎么理解语言？**

20 世纪 50 年代到 70 年代，人们对自然语言处理的认识都局限在人类学习语言的方式上，用了二十多年时间苦苦探寻让计算机理解语言的方法，最终却一无所获。

当时的学术界普遍认为，要让计算机处理自然语言必须先让其理解语言，因此分析语句和获取语义成为首要任务，而这主要依靠语言学家人工总结文法规则来实现。特别是 20 世纪 60 年代，基于乔姆斯基形式语言（Chomsky Formal languages）的编译器取得了很大进展，更加鼓舞了研究者通过概括语法规则来处理自然语言的信心。

<img src="/assets/img/nnlm-to-bert/chomsky.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 200px">

<center>图 1-1 诺姆·乔姆斯基（Noam Chomsky）</center>

但是与规范严谨的程序语言不同，自然语言复杂又灵活，是一种上下文有关文法（Context-Sensitive Grammars，CSGs），因此仅靠人工编写文法规则根本无法覆盖，而且随着编写的规则数量越来越多、形式越来越复杂，规则与规则之间还可能会存在矛盾。因此这一阶段自然语言处理的研究可以说进入了误区。

**第二阶段：只要看的足够多，就能处理语言**

正如人类是通过空气动力学而不是简单模仿鸟类造出了飞机，计算机处理自然语言也未必需要理解语言。

<img src="/assets/img/nnlm-to-bert/computer_learn_language.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 550px">

<center>图 1-2 处理语言需要先理解语言吗？</center>

20 世纪 70 年代，随着统计语言学的提出，基于数学模型和统计方法的自然语言处理方法开始兴起。当时的代表性方法是“通信系统加隐马尔可夫模型”，其输入和输出都是一维且保持原有次序的符号序列，可以处理语音识别、词性分析等任务，但是这种方法在面对输出为二维树形结构的句法分析以及符号次序有很大变化的机器翻译等任务时就束手无策了。

20 世纪 80 年代以来，随着硬件计算能力的提高以及海量互联网数据的出现，越来越多的统计机器学习方法被应用到自然语言处理领域，例如一些研究者引入基于有向图的统计模型来处理复杂的句法分析任务。2005 年 Google 公司基于统计方法的翻译系统更是全面超过了基于规则的 SysTran 系统。

<img src="/assets/img/nnlm-to-bert/hinton.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 200px">

<center>图 1-3 杰弗里·辛顿（Geoffrey Hinton）</center>

2006 年，随着辛顿（Hinton）证明深度信念网络（Deep Belief Networks，DBN）可以通过逐层预训练策略有效地进行训练，基于神经网络和反向传播算法（Back Propagation）的深度学习方法开始兴起。许多之前由于缺乏数据、计算能力以及有效优化方法而被忽视的神经网络模型得到了复兴。例如 1997 年就已提出的长短时记忆网络（Long Short Term Memory，LSTM）模型在重新被启用后在许多任务上大放异彩。

> **延伸**
>
> 即使在 Transformer 模型几乎“一统江湖”的今天，LSTM 模型依然占有一席之地。2024 年 5 月 8 日，LSTM 提出者和奠基者 Sepp Hochreiter 公布了 LSTM 模型的改良版本——xLSTM，在性能和扩展方面都得到了显著提升。论文的所属机构中还出现了一家叫做 NXAI 的公司，Sepp Hochreiter 表示：“借助 xLSTM，我们缩小了与现有最先进大语言模型的差距。借助 NXAI，我们已开始构建欧洲自己的大语言模型。”
{: .block-tip }

随着越来越多研究者将注意力转向深度学习方法，诸如卷积神经网络（Convolutional Neural Networks，CNN）等模型被广泛地应用到各种自然语言处理任务中。2017 年，Google 公司提出了 Attention 注意力模型，论文中提出的 Transformer 结构更是引领了后续神经网络语言模型的发展。

得益于抛弃了让计算机简单模仿人类的思路，这一阶段自然语言处理研究出现了蓬勃发展。今天可以说已经没有人再会质疑统计方法在自然语言处理上的可行性。

## 1.2 统计语言模型发展史

要让计算机处理自然语言，首先需要为自然语言建立数学模型，这种模型被称为“统计语言模型”，其核心思想是判断一个文字序列是否构成人类能理解并且有意义的句子。这个问题曾经困扰了学术界很多年。

### 统计语言模型

20 世纪 70 年代之前，研究者尝试从文字序列是否合乎文法、含义是否正确的角度来建立语言模型。最终，随着人工编写出的规则数量越来越多、形式越来越复杂，对语言模型的研究陷入瓶颈。直到 20 世纪 70 年代中期，IBM 实验室的贾里尼克（Jelinek）为了研究语音识别问题换了一个思路，用一个简单的统计模型就解决了这个问题。

<img src="/assets/img/nnlm-to-bert/jelinek.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 200px">

<center>图 1-4 佛里德里克·贾里尼克（Frederick Jelinek）</center>

贾里尼克的想法是要判断一个文字序列 $w_1,w_2,…,w_n$ 是否合理，就计算这个句子 $S$ 出现的概率 $P(S)$，出现概率越大句子就越合理：

$$
P(S) = P(w_1,w_2,...,w_n)\\
= P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)...P(w_n|w_1,w_2,...,w_{n-1})\tag{1.1}
$$

其中，词语 $w_n$ 出现的概率取决于在句子中出现在它之前的所有词（理论上也可以引入出现在它之后的词语）。但是，随着文本长度的增加，条件概率 $P(w_n\mid w_1,w_2,…,w_{n-1})$ 会变得越来越难以计算，因而在实际计算时会假设每个词语 $w_i$ 仅与它前面的 $N−1$ 个词语有关，即：

$$
P(w_i|w_1,w_2,...,w_{i-1}) = P(w_i|w_{i-N+1},w_{i-N+2},...,w_{i-1})
$$

这种假设被称为马尔可夫（Markov）假设，对应的语言模型被称为 $N$ 元（$N$-gram）模型。例如当 $N=2$ 时，词语 $w_i$ 出现的概率只与它前面的词语 $w_{i−1}$ 有关，被称为二元（Bigram）模型；而 $N=1$ 时，模型实际上就是一个上下文无关模型。由于 $N$ 元模型的空间和时间复杂度都几乎是 $N$ 的指数函数，因此实际应用中比较常见的是取 $N=3$ 的三元模型。

> **延伸**
>
> 即使是使用三元、四元甚至是更高阶的语言模型，依然无法覆盖所有的语言现象。在自然语言中，上下文之间的关联性可能跨度非常大，例如从一个段落跨到另一个段落，这是马尔可夫假设解决不了的。此时就需要使用 LSTM、Transformer 等模型来捕获词语之间的远距离依赖（Long Distance Dependency）了。
{: .block-tip }

### NNLM 模型

2003 年，本吉奥（Bengio）提出了神经网络语言模型（Neural Network Language Model，NNLM)。可惜它生不逢时，由于神经网络在当时并不被人们看好，在之后的十年中 NNLM 模型都没有引起很大关注。

<img src="/assets/img/nnlm-to-bert/bengio.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 200px">

<center>图 1-5 约书亚·本吉奥（Yoshua Bengio）</center>

直到 2013 年，随着越来越多的研究者使用深度学习模型来处理自然语言，NNLM 模型才被重新发掘，并成为使用神经网络建模语言的经典范例。NNLM 模型的思路与统计语言模型保持一致，它通过输入词语前面的 $N−1$ 个词语来预测当前词。模型结构如图 1-6 所示。

<img src="/assets/img/nnlm-to-bert/nnlm.jpg" width="600px" style="display:block; margin:auto;">

<center>图 1-6 神经网络语言模型（NNLM）的结构</center>

具体来说，NNLM 模型首先从词表 $C$ 中查询得到前面 $N−1$ 个词语对应的词向量 $C(w_{t-n+1}),…,C(w_{t-2}),C(w_{t-1})$，然后将这些词向量拼接后输入到带有激活函数的隐藏层中，通过 $\text{Softmax}$ 函数预测当前词语的概率。特别地，包含所有词向量的词表矩阵 $C$ 也是模型的参数，需要通过学习获得。因此 NNLM 模型不仅能够能够根据上文预测当前词语，同时还能够给出所有词语的词向量（Word Embedding）。

### Word2Vec 模型

真正将神经网络语言模型发扬光大的是 2013 年 Google 公司提出的 Word2Vec 模型。Word2Vec 模型提供的词向量在很长一段时间里都是自然语言处理方法的标配，即使是后来出现的 Glove 模型也难掩它的光芒。

Word2Vec 的模型结构和 NNLM 基本一致，只是训练方法有所不同，分为 CBOW (Continuous Bag-of-Words) 和 Skip-gram 两种，如图 1-7 所示。

<img src="/assets/img/nnlm-to-bert/word2vec.jpg" width="700px" style="display:block; margin:auto;">

<center>图 1-7 Word2Vec 模型的训练方法</center>

其中 CBOW 使用周围的词语 $w(t-2),w(t-1),w(t+1),w(t+2)$ 来预测当前词 $w(t)$，而 Skip-gram 则正好相反，它使用当前词 $w(t)$ 来预测它的周围词语。

可以看到，与严格按照统计语言模型结构设计的 NNLM 模型不同，Word2Vec 模型在结构上更加自由，训练目标也更多地是为获得词向量服务。特别是同时通过上文和下文来预测当前词语的 CBOW 训练方法打破了语言模型“只通过上文来预测当前词”的固定思维，为后续一系列神经网络语言模型的发展奠定了基础。

然而，有一片乌云一直笼罩在 Word2Vec 模型的上空——多义词问题。一词多义是语言灵活性和高效性的体现，但是 Word2Vec 模型却无法处理多义词，一个词语无论表达何种语义，Word2Vec 模型都只能提供相同的词向量，即将多义词编码到了完全相同的参数空间。实际上在 20 世纪 90 年代初，雅让斯基（Yarowsky）就给出了一个简洁有效的解决方案——运用词语之间的互信息（Mutual Information）。

<img src="/assets/img/nnlm-to-bert/yarowsky.jpg" alt="masked_modeling" style="display: block; margin: auto; width: 200px">

<center>图 1-8 大卫·雅让斯基（David Yarowsky）</center>

具体来说，对于多义词，可以使用文本中与其同时出现的互信息最大的词语集合来表示不同的语义。例如对于“苹果”，当表示水果时，周围出现的一般就是“超市”、“香蕉”等词语；而表示“苹果公司”时，周围出现的一般就是“手机”、“平板”等词语，如图 1-9 所示。

<img src="/assets/img/nnlm-to-bert/polysemy_problem.jpg" width="400px" style="display:block; margin:auto;">

<center>图 1-9 运用互信息解决多义词问题</center>

因此，在判断多义词究竟表达何种语义时，只需要查看哪个语义对应集合中的词语在上下文中出现的更多就可以了，即通过上下文来判断语义。

> **延伸**
>
> 1948 年，香农（Claude Elwood Shannon）在他著名的论文《通信的数学原理》中提出了“信息熵”（Information Entropy）的概念，解决了信息的度量问题，并且量化出信息的作用。上面提到的互信息就来自于信息论。如果你对此感兴趣，可以阅读[《信息的度量和作用：信息论基本概念》](https://xiaosheng.blog/2017/03/09/how-to-measure-information)。
{: .block-tip }

后来自然语言处理的标准流程就是先将 Word2Vec 模型提供的词向量作为模型的输入，然后通过 LSTM、CNN 等模型结合上下文对句子中的词语重新进行编码，以获得包含上下文信息的词语表示。

### ELMo 模型

为了更好地解决多义词问题，2018 年研究者提出了 ELMo 模型（Embeddings from Language Models）。与 Word2Vec 模型只能提供静态词向量不同，ELMo 模型会根据上下文动态地调整词语的词向量。

具体来说，ELMo 模型首先对语言模型进行预训练，使得模型掌握编码文本的能力；然后在实际使用时，对于输入文本中的每一个词语，都提取模型各层中对应的词向量拼接起来作为新的词向量。ELMo 模型采用双层双向 LSTM 作为编码器，如图 1-10 所示，从两个方向编码词语的上下文信息，相当于将编码层直接封装到了语言模型中。

<img src="/assets/img/nnlm-to-bert/elmo.jpg" width="600px" style="display:block; margin:auto;">

<center>图 1-10 ELMo 模型的结构</center>

训练完成后 ELMo 模型不仅学习到了词向量，还训练好了一个双层双向的 LSTM 编码器。对于输入文本中的词语，可以从第一层 LSTM 中得到包含句法信息的词向量，从第二层 LSTM 中得到包含语义信息的词向量，最终通过加权求和得到每一个词语最终的词向量。

但是 ELMo 模型存在两个缺陷：首先它使用 LSTM 模型作为编码器，而不是当时已经提出的编码能力更强的 Transformer 模型；其次 ELMo 模型直接通过拼接来融合双向抽取特征的做法也略显粗糙。

不久之后，将 ELMo 模型中的 LSTM 更换为 Transformer 的 GPT 模型就出现了。但是 GPT 模型再次追随了 NNLM 的脚步，只通过词语的上文进行预测，这在很大程度上限制了模型的应用场景。例如对于文本分类、阅读理解等任务，如果不把词语的下文信息也嵌入到词向量中就会白白丢掉很多信息。

### BERT 模型

2018 年底随着 BERT 模型（Bidirectional Encoder Representations from Transformers）的出现，这一阶段神经网络语言模型的发展终于出现了一位集大成者，发布时 BERT 模型在 11 个任务上都取得了最好性能。

BERT 模型采用和 GPT 模型类似的两阶段框架，首先对语言模型进行预训练，然后通过微调来完成下游任务。但是，BERT 不仅像 GPT 模型一样采用 Transformer 作为编码器，而且采用了类似 ELMo 模型的双向语言模型结构，如图 1-11 所示。因此 BERT 模型不仅编码能力强大，而且对各种下游任务，BERT 模型都可以通过简单地改造输出部分来完成。

<img src="/assets/img/nnlm-to-bert/bert.jpg" width="300px" style="display:block; margin:auto;">

<center>图 1-11 BERT 模型的结构</center>

但是 BERT 模型的优点同样也是它的缺陷，由于 BERT 模型采用双向语言模型结构，因而无法直接用于生成文本。

可以看到，从 2003 年 NNLM 模型提出时的无人问津，到 2018 年底 BERT 模型横扫自然语言处理领域，神经网络语言模型的发展也经历了一波三折。在此期间，研究者一直在不断地对前人的工作进行改进，这才取得了 BERT 模型的成功。BERT 模型的出现并非一蹴而就，它不仅借鉴了 ELMo、GPT 等模型的结构与框架，而且延续了 Word2Vec 模型提出的 CBOW 训练方式的思想，可以看作是这一阶段语言模型发展的集大成者。

在 BERT 模型取得成功之后，研究者并没有停下脚步，在 BERT 模型的基础上又提出了诸如 MASS、ALBERT、RoBERTa 等改良模型。其中具有代表性的就是微软提出的 UNILM 模型（UNIfied pretrained Language Model），它把 BERT 模型的 MASK 机制运用到了一个很高的水平，如图 1-12 所示。

<img src="/assets/img/nnlm-to-bert/unilm.jpg" width="700px" style="display:block; margin:auto;">

<center>图 1-12 UNILM 模型的结构</center>

具体来说，UNILM 模型通过给 Transformer 中的 Self-Attention 机制添加不同的 MASK 矩阵，在不改变 BERT 模型结构的基础上同时实现了双向、单向和序列到序列（Sequence-to-Sequence，Seq2Seq）语言模型，是一种对 BERT 模型进行扩展的优雅方案。

### 大语言模型

除了优化模型结构，研究者发现扩大模型规模也可以提高性能。在保持模型结构以及预训练任务基本不变的情况下，仅仅通过扩大模型规模就可以显著增强模型能力，尤其当规模达到一定程度时，模型甚至展现出了能够解决未见过复杂问题的涌现（Emergent Abilities）能力。例如 175B 规模的 GPT-3 模型只需要在输入中给出几个示例，就能通过上下文学习（In-context Learning）完成各种小样本（Few-Shot）任务，而这是 1.5B 规模的 GPT-2 模型无法做到的。

<img src="/assets/img/nnlm-to-bert/timeline_of_llm.jpg" width="900px" style="display:block; margin:auto;">

<center>图 1-13 近年来发布的一些大语言模型（10B 规模以上）</center>

在规模扩展定律（Scaling Laws）被证明对语言模型有效之后，研究者基于 Transformer 结构不断加深模型深度，构建出了许多大语言模型，如图 1-13 所示。

一个标志性的事件是 2022 年 11 月 30 日 OpenAI 公司发布了面向普通消费者的 ChatGPT 模型（Chat Generative Pre-trained Transformer），它能够记住先前的聊天内容真正像人类一样交流，甚至能撰写诗歌、论文、文案、代码等。发布后，ChatGPT 模型引起了巨大轰动，上线短短 5 天注册用户数就超过 100 万。2023 年一月末，ChatGPT 活跃用户数量已经突破 1 亿，成为史上增长最快的消费者应用。

下面本章将按照模型规模介绍一些可供开发者使用的大语言模型。首先是数百亿参数的大语言模型：

- Flan-T5（11B）：指令微调（Instruction Tuning）研究领域的代表性模型，通过扩大任务数量、扩大模型规模以及在思维链提示（Chain-of-Thought Prompting）数据上进行微调探索了指令微调技术的应用；
- CodeGen 以及  CodeGen2（11B）：为生成代码而设计的自回归（Autoregressive）语言模型，是探索大语言模型代码生成能力的一个代表性模型；
- mT0（13B）：多语言（Multilingual）大语言模型的代表，使用多语言提示在多语言任务上进行了微调；
- Baichuan 以及 Baichuan-2（7B）：百川智能公司开发的大语言模型，支持中英双语，在多个中英文基准测试上取得优异性能；
- PanGu-$\alpha$（13B）：华为公司开发的中文大语言模型，在零样本（Zero-Shot）和小样本（Few-Shot）设置下展现出了优异的性能；
- Qwen（72B）：阿里巴巴公司开源的多语言大模型，在语言理解、推理、数学等方面均展现出了优秀的模型能力，还为代码、数学和多模态设计了专业化版本 Code-Qwen、Math-Qwen、Qwen-VL 等可供用户使用；
- LLaMA 以及 LLaMA-2（65B）：在一系列指令遵循（Instruction Following）任务中展现出卓越性能。由于 LLaMA 模型的开放性和有效性，吸引了许多研究者在其之上指令微调或继续预训练不同的模型版本，例如 Stanford Alpaca 模型、Vicuna 模型等，如图 1-14 所示。
- Mixtral（46.7B）：稀疏混合专家模型架构的大语言模型，这也是较早对外公开的 MoE 架构的语言模型，其处理速度和资源消耗与 12.9B 参数的模型相当，在 MT-bench 基准上取得了与 GPT-3.5 相当的性能表现；

<img src="/assets/img/nnlm-to-bert/llama_family.jpg" width="1000px" style="display:block; margin:auto;">

<center>图 1-14 LLaMA 模型家族</center>

然后是数千亿计参数规模的大语言模型：

- OPT（175B）以及指令微调版本 OPT-IML：致力于开放共享，使得研究者可以对大规模模型进行可复现的研究；
- BLOOM 以及 BLOOMZ（176B）：跨语言泛化（Cross-Lingual Generalization）研究领域的代表性模型，具有多语言建模的能力；
- GLM：双语大语言模型，其小规模中文聊天版本 ChatGLM2-6B 在中文任务研究中十分流行，在效率和容量方面有许多改进，支持量化（Quantization）、32K 长度的上下文、快速推理等。

> **相关**
>
> 如果你对提示（Prompting）、指令微调（Instruction Tuning）等专业术语不熟悉也不用着急，本教程会在第十三章《Prompting 情感分析》以及第十四章《使用大语言模型》中进行详细介绍。
{: .block-warning }

当然，对于普通开发者，一种更简单的方式是直接调用大语言模型接口，这样就不需要在本地搭建环境部署模型。例如通过 OpenAI 公司提供的接口就可以调用 GPT-3、GPT-3.5、GPT-4 等一系列的 GPT 模型，其中一些还支持通过接口进行微调。

> **延伸**
>
> 如果你对如何通过接口调用 GPT 模型感兴趣，可以阅读[《ChatGPT 教程 (Python 调用 OpenAI API)》](https://xiaosheng.blog/2023/05/04/how-to-use-chatgpt)。
{: .block-tip }

## 1.3 小结

可以看到，自然语言处理的发展并非一帆风顺，期间也曾走入歧路而停滞不前，正是一代又一代研究者的不懈努力才使得该领域持续向前发展并取得了许多令人印象深刻的成果。如今预训练语言模型、大语言模型在学术界和工业界都获得了广泛的应用，深刻地改变着我们的生活，我们需要明白这些成功并非一蹴而就，而是“站在巨人的肩膀上”。

## 参考

[[1]](https://book.douban.com/subject/26163454/) 吴军.2014.数学之美 （第二版）.人民邮电出版社  
[[2]](https://book.douban.com/subject/26708119/) 周志华.2016.机器学习.清华大学出版社  
[[3]](https://zhuanlan.zhihu.com/p/49271699) 张俊林.从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史  
[[4]](https://github.com/RUCAIBox/LLMSurvey) 赵鑫等.A Survey of Large Language Models
