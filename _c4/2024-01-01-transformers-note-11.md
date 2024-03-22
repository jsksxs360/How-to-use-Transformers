---


## title: 第十四章：大模型简介
author: SHENG XU
date: 2024-01-01
category: NLP
layout: post

！！！这行不要删：请保留两级目录，使用（##）和（###）开头。
# **摘要**
语言本质上是一个受语法规则控制的复杂的人类表达系统。开发强大的人工智能 (AI) 算法来理解和掌握语言提出了重大挑战。作为一种主要方法，语言建模在过去二十年中被广泛研究用于语言理解和生成，从统计语言模型发展到神经语言模型。最近，通过在大规模语料库上预训练 Transformer 模型提出了预训练语言模型（PLM），在解决各种自然语言处理（NLP）任务方面表现出了强大的能力。由于研究人员发现模型缩放可以提高模型容量，因此他们通过将参数比例增加到更大的尺寸来进一步研究缩放效果，**当参数规模超过一定水平时，这些扩大的语言模型不仅实现了显着的性能提升，而且还表现出了一些小规模语言模型（例如 BERT）中不存在的特殊能力（例如上下文学习）** 。为了区分不同参数规模的语言模型，研究界为具有相当规模（例如，包含数百或数千亿个参数）的 PLM 创建了术语“大型语言模型”(LLM)。近年来，学界和工业界对LLM的研究取得了很大进展，其中一个显着进展是ChatGPT（基于LLM开发的强大AI聊天机器人）的推出。LLMs的技术发展对整个人工智能社区产生了重要影响，这将彻底改变我们开发和使用人工智能算法的方式。考虑到这种快速的技术进步，**我们主要介绍背景、主要发现和主流技术来回顾LLMs的最新进展。**我们特别关注LLM的四个主要方面，即预训练、适应调优、利用和能力评估。此外，我们还总结了发展LLMs的可用资源，并讨论了未来方向的剩余问题。这项调查提供了有关LLMs的最新文献综述，这对研究人员和工程师来说都是有用的资源。
# **简介**
从技术上讲，**语言建模（LM）是提高机器语言智能的主要方法之一。一般来说，LM 的目标是对单词序列的生成可能性进行建模，从而预测未来（或缺失）标记的概率。** LM的研究受到文献的广泛关注，可分为四个主要发展阶段：
· 统计语言模型（SLM）基本思想是基于马尔可夫假设构建单词预测模型。具有固定上下文长度 n 的 SLM 也称为 n-gram 语言模型，例如二元模型和三元模型。 SLM 已广泛应用于增强信息检索 (IR) 和自然语言处理 (NLP) 中的任务性能。**然而，它们经常遭受维数灾难：由于需要估计指数数量的转移概率，因此很难准确估计高阶语言模型。**因此，引入了专门设计的平滑策略，例如退避估计和Good-Turing估计来缓解数据稀疏问题。
· 神经语言模型（NLM）。 NLM 通过神经网络（例如循环神经网络（RNN））来表征单词序列的概率。作为一个显着的贡献，中的工作引入了单词的分布式表示的概念，并建立了以聚合上下文特征（即分布式单词向量）为条件的单词预测函数。通过扩展学习单词或句子的有效特征的思想，开发了一种通用的神经网络方法来为各种 NLP 任务构建统一的解决方案 。
· 预训练语言模型 (PLM)。早期ELMo 被提出通过首先预训练双向 LSTM (biLSTM) 网络（而不是学习固定的单词表示），然后根据特定的下游微调 biLSTM 网络来捕获上下文感知的单词表示，这些预先训练的上下文感知单词表示作为通用语义特征非常有效。这项工作的成功，启发后续设定了“预训练和微调”的学习范式。遵循这一范式，已经开展了大量关于 PLM 的研究，引入了不同的架构 。
· 大型语言模型（LLM）。研究人员发现，扩展 PLM（例如扩展模型大小或数据大小）通常会提高下游任务的模型能力（即遵循扩展定律 ）。许多研究通过训练更大的 PLM（例如 175B 参数 GPT-3 和 540B 参数 PaLM）来探索性能极限。**尽管缩放主要在模型大小上进行（具有相似的架构和预训练任务），但这些大型 PLM 显示出与小型 PLM（例如 330M 参数 BERT 和 1.5B 参数 GPT-2）不同的行为，并显示出令人惊讶的能力（称为解决一系列复杂任务的新兴能力）。**
我们首先强调 LLM 和 PLM 之间的**三个主要区别：首先，LLMs展示了一些令人惊讶的新兴能力，这些能力在以前的小型 PLM 中可能无法观察到。其次，LLMs将彻底改变人类开发和使用人工智能算法的方式。第三，LLMs的发展不再明确区分研究和工程。**
如今，ChatGPT 和 GPT-4 的出现引发了人们对通用人工智能 (AGI) 可能性的重新思考。 OpenAI 发表了一篇题为“Planning for AGI and Beyond”的技术文章，讨论了实现 AGI 的短期和长期计划。最近的一篇论文认为 GPT-4 可能被视为 AGI 系统的早期版本。**LLMs的快速发展正在给人工智能的研究领域带来革命性的变化。在NLP领域，LLM可以（在某种程度上）充当通用语言任务求解器，并且研究范式已经转向使用LLM。**
**尽管取得了进展和影响，LLMs的基本原则仍然没有得到很好的探索。首先，令人费解的是为什么新兴能力出现在 LLM 中，而不是较小的 PLM 中**。**其次，研究界很难培养有能力的LLMs。第三，使LLMs与人类价值观或偏好保持一致具有挑战性。**
机遇与挑战并存，LLMs的研究与发展需要更多的关注。**为了让大家对LLM有一个基本的了解，本次调查从预训练（如何预训练有能力的LLM）、适应（如何有效地适应预训练的LLM）四个主要方面对LLM的最新进展进行了文献综述。以便更好地使用）、利用率（如何利用LLMs来解决各种下游任务）和能力评估（如何评估LLMs的能力和现有的实证结果）。**

## **LLMs的背景**
通常，大型语言模型（LLM）是指包含数千亿参数的 Transformer 语言模型，这些模型在海量文本数据上进行训练，例如 GPT-3 、PaLM 和 LLaMA。**LLMs表现出强大的理解自然语言和解决复杂任务（通过文本生成）的能力**。为了快速了解LLM的工作原理，本部分介绍了LLM的基本背景，包括缩放法则、涌现能力和关键技术。
LLMs的拓展法则：目前，LLM 主要构建在 Transformer 架构上，其中多头注意力层堆叠在非常深的神经网络中。现有的LLMs采用类似的 Transformer 架构和预训练目标（例如语言建模）作为小型语言模型。然而，LLMs显着扩展了模型大小、数据大小和总计算量（放大数量级）。广泛的研究表明，扩展可以很大程度上提高LLMs的模型能力。因此，建立一种定量方法来表征缩放效应是有用的。接下来，我们介绍 Transformer 语言模型的两个代表性缩放定律。
1. KM缩放定律：**2020 年OpenAI团队首先提出对模型性能与三个主要因素的幂律关系进行建模，即模型大小（N）、数据集大小（D）和训练计算量（C），表明模型性能对这三个因素有很强的依赖关系。**
2. Chinchilla缩放定律：Hoffmann 等人提出了一种缩放法则的替代形式来指导LLMs的计算优化训练。
LLMs的涌现能力：**LLM的涌现能力被正式定义为“小模型中不存在但在大模型中出现的能力”，这是LLM区别于以往PLM的最显着特征之一**。就是当规模达到一定水平时，性能显着高于随机性能。以此类推，这种涌现模式与物理学中的相变现象有着密切的联系。
介绍一下LLMs的三种典型涌现能力以及具备这种能力的代表性模型：
1. 情境学习：上下文学习（ICL）能力由 GPT-3 正式引入：假设语言模型已提供自然语言指令和/或多个任务演示，它可以生成测试实例的预期输出通过完成输入文本的单词序列，无需额外的训练或梯度更新。
2. 遵循指令：通过对通过自然语言描述格式化的多任务数据集的混合进行微调（称为指令调整），LLM 在以指令形式描述的看不见的任务上表现良好 。
3. 逐步推理：对于小型语言模型，通常很难解决涉及多个推理步骤的复杂任务，例如数学文字问题。相比之下，通过思想链（CoT）提示策略，LLMs可以利用涉及中间推理步骤以获得最终答案的提示机制来解决此类任务。
LLMs的关键技术：
1. 缩放：正如前面部分所讨论的，Transformer 语言模型存在明显的缩放效应：更大的模型/数据大小和更多的训练计算通常会导致模型容量的提高。
2. 训练：模型规模太大，需要分布式训练算法来学习LLM的网络参数，其中常常联合使用各种并行策略。为了支持分布式训练，已经发布了几个优化框架来促进并行算法的实现和部署，例如 DeepSpeed 和 Megatron-LM 。
3. 能力激发：经过大规模语料库的预训练后，LLMs被赋予了作为通用任务解决者的潜在能力作为技术方法，但这种能力有时候不会显示，设计合适的任务指令或特定的情境学习策略来激发这种能力是有用的。例如，思想链提示已被证明可以通过包含中间推理步骤来解决复杂的推理任务。此外，我们可以对用自然语言表达的任务描述对LLM进行指令调优，以提高LLM在未见过的任务上的泛化能力。
4. 对准调整：由于LLMs经过训练来捕获预训练语料库的数据特征（包括高质量和低质量数据）。这个过程中它们可能会对人类有偏见，有必要使LLMs与人类价值观保持一致。InstructGPT设计一种有效的调整方法，使LLMs能够遵循预期的指示
5. 工具操作：LLMs被训练为大量纯文本语料库的文本生成器，因此在不能最好地以文本形式表达的任务（例如数值计算）上表现不佳。此外，它们的能力也仅限于预训练数据，例如无法捕获最新信息。我们利用外部工具来弥补LLMs的缺陷，ChatGPT也启用了一些外部插件。
## **GPT系列型号的技术演变**
GPT模型的基本原理是通过语言建模将世界知识压缩为仅解码器的Transformer模型，使其能够恢复（或记忆）整体知识的语义并作为通用任务求解器。成功的两个关键点是（I）训练可以准确预测下一个单词的仅解码器 Transformer 语言模型，以及（II）扩大语言模型的规模。总体而言，OpenAI对LLMs的研究大致可以分为以下几个阶段：
早期：2017年Google推出了Transformer模型，OpenAI团队很快将他们的语言建模工作适应了这种新的神经网络架构。他们于2018年发布了第一个GPT模型，即GPT-1。
遵循与 GPT-1 类似的架构，GPT-2 使用大型网页数据集 WebText 进行训练，它试图通过无监督语言建模来执行任务，而不使用标记数据进行显式微调。为了激发该方法，他们引入了一种用于多任务解决的概率形式，即 p(output|input，task)。对这一主张的基本理解是，每个（NLP）任务都可以被视为基于全局文本子集的单词预测问题。因此，如果无监督语言模型经过训练具有足够的能力来恢复全局文本，那么它就能够解决各种任务。
GPT-3 正式引入了上下文学习（ICL）的概念。通过 ICL，LLM 的预训练和利用收敛到相同的语言建模范式：预训练根据上下文预测以下文本序列，而 ICL 预测正确的任务解决方案，该解决方案也可以格式化为文本序列，给出任务描述和演示。
由于能力强大，GPT3 已成为为 OpenAI 开发能力更强的LLMs的基础模型。总体而言，OpenAI探索了两种主要方法来进一步改进GPT-3模型，即代码数据训练和与人类偏好一致
OpenAI已经实现了两个重要的里程碑，即ChatGPT和GPT-4。
ChatGPT 的训练方式与 InstructGPT 类似，但也有训练在数据收集设置方面的差异：人类生成的对话（同时扮演用户和 AI 的角色）以对话格式与 InstructGPT 数据集相结合，用于训练 ChatGPT。拥有丰富的知识储备、数学问题推理能力、在多轮对话中准确追踪上下文、符合人类价值观以确保安全使用。
GPT-4。作为另一个显着进展：将文本输入扩展到多模态信号。总体而言，GPT-4 解决复杂任务的能力比 GPT-3.5 更强，在许多评估任务上表现出较大的性能提升.
尽管取得了巨大进步，这些高级LLMs仍然存在局限性。例如，在某些特定背景下产生带有事实错误的幻觉或潜在的危险反应。
# **LLMs的资源**
开发或复制LLMs绝非易事。一种可行的方法是向现有LLMs学习经验，并重用公开资源进行增量开发或实验研究。在本节中，我们简要总结了用于开发 LLM 的公开可用资源，包括模型检查点（或 API）、语料库和库。
## 当前可用模型和APIS
由于参数规模是使用LLM需要考虑的关键因素，因此我们将这些公共模型分为两个规模级别（即百亿参数和千亿参数）
百亿参数模型：该类别中的大多数模型的参数规模范围为 10B 到 20B，除了 LLaMA、NLLB和 Falcon。该范围内的其他模型包括 mT5 、PanGu-α、T0、GPTNeoX-20B、CodeGen、UL2 、Flan-T5 和 mT0。其中，Flan-T5可以作为指令调优研究的首要模型，因为它从三个方面探索指令调优：增加任务数量、缩放模型大小和微调具有思路链提示的数据。对于多语言任务，mT0可能是一个很好的候选模型，它已经针对具有多语言提示的多语言任务进行了微调。此外，PanGu-α 在零样本或少样本设置下的中文下游任务中表现出良好的性能。作为一种流行的LLM，LLaMA，其包含的参数大约是其他模型的五倍，在与指令跟踪相关的任务中表现出了卓越的性能。
千亿参数模型：这类公开的模型比较少。例如，OPT、OPT-IML、BLOOM和BLOOMZ的参数数量与GPT-3几乎相同。其中，OPT以及指令调整版本OPT-IML特别鼓励开放共享，旨在使研究人员能够大规模地进行可重复的研究。对于跨语言泛化的研究，由于具有多语言建模任务的能力，BLOOM和BLOOMZ可以用作基础模型。作为双语LLM，GLM还提供了流行的小型中文聊天模型ChatGLM2-6B，该模型在效率和容量上都有很多改进（例如量化、32K长度的上下文、推理速度快）。
LLaMA 模型族：LLaMA 模型集合于 2023 年 2 月推出，包括四种大小。自发布以来，LLaMA 引起了研究界和工业界的广泛关注。许多研究人员通过指令调整或持续预训练来扩展 LLaMA 模型。特别是，由于计算成本相对较低，结构调整 LLaMA 已成为开发定制或专用模型的主要方法。为了有效地适应非英语语言的LLaMA模型，通常需要扩展原始词汇（主要在英语语料库上训练）或使用目标语言的指令或数据对其进行微调。在这些扩展模型中，Stanford Alpaca 是第一个基于 LLaMA 进行微调的开放指令跟随模型。Vicuna 是另一种流行的 LLaMA 变体，根据从 ShareGPT 16 收集的用户共享对话进行训练。Vicuna 在多模态语言模型中更受青睐，这导致了多种流行模型的出现，包括 LLaVA、MiniGPT4 、InstructBLIP和 PandaGPT 。
LLMs的公共APIs:API 不是直接使用模型副本，而是为普通用户提供了更方便的使用 LLM 的方式，而无需在本地运行模型。OpenAI为GPT-3系列中的模型提供了七大接口：ada、babbage、curie、davinci、text-ada-001、text-babbage-001和text-curie -001。此外，GPT-3.5系列包括一个基本模型代码davinci-002和三个增强版本，即text-davinci-002、text-davinci-003和gpt-3.5-turbo-0301。值得注意的是，gpt-3.5-turbo-0301是调用ChatGPT的接口。最近，OpenAI还发布了GPT-4的相应API，包括gpt-4、gpt-4-0314、gpt-4-32k和gpt-4-32k-0314。
## **常用语料库**
与早期的 PLM 相比，LLM 包含大量参数，需要更多涵盖广泛内容的训练数据.我们将简要总结几个广泛使用的用于训练LLMs的语料库。主要包括六大类：书籍、CommonCrawl、Reddit 链接、维基百科、代码等。
书籍主要是BookCorpus 用于小规模模型训练和古腾堡计划语料库。CommonCrawl 是最大的开源网络爬虫数据库之一，包含PB级数据量，已被广泛用作现有LLMs的训练数据.
Reddit 是一个社交媒体平台，用户可以提交链接和文本帖子，其他人可以通过“赞成票”或“反对票”对其进行投票。高度赞扬的帖子通常被认为是有用的，可以用来创建高质量的数据集。OpenWebText语料库是由以上高投票链接组成的。PushShift.io是另一个从Reddit 中提取的实时更新的语料库。
维基百科语料库是一个在线百科全书，包含大量关于不同主题的高质量文章。
Pile 数据集是一个大规模、多样化的开源文本数据集，由来自多个来源的超过 800GB 的数据组成，包括书籍、网站、代码、科学论文和社交媒体平台。
## **图书馆资源**
主要包括:Transformers库，DeepSpeed 深度学习优化库，Megatron-LM，JAX，Colossal-AI，BMTrain，FastMoE等。
除了上述库资源外，现有的深度学习框架（例如，PyTorch 、TensorFlow、MXNet、PaddlePaddle 、MindSpore 和 OneFlow也提供了支持用于并行算法，通常用于训练大型模型。
## 预训练

预训练是LLMs获取强大能力的基础。由于LLMs的预训练过程通常具有资源成本耗费巨大，训练时间长等特点，所以如何更有效率地完成模型的预训练过程显得十分重要，其中涉及到很多因素如数据的收集与处理、模型结构的选择、训练技巧等。下面将围绕这些因素进行介绍。

### 数据收集

现有的LLMs所用语料可大概分为两部分：通用数据（General data）和专业数据（Specialized data）。通用数据可以增强LLMs的语言建模能力和泛化能力，而专业数据则有助于增强LLMs在特定任务上的表现。

通用数据大致分为三部分：

1. Webpages:包含来自互联网上的各种数据，但所得到的数据质量良莠不齐，需要进一步的数据清洗。
2. Conversation text:其有助于增强LLMs在对话或者问答领域的能力。在具体处理中由于对话数据通常涉及到多个参与者，所以将其划分为多个子对话进行使用。一些研究发现过度使用对话数据可能使得LLMs错误地将一些陈述性指令和直接疑问句视为开头，从而影响指令（instruction）的效果。
3. Books:与上述两种语料相比，其包含了更多的长文本，有助于LLMs获得更多的语言知识，增强连贯文本的生成能力。常用数据集有Books3、Bookcorpus2。

具体数据也大致分为三部分：

1. Multilingual text:多语言文本的使用有助于增强LLMs的多语言理解与生成能力，有助于LLMs更好的处理多语言任务如文本翻译，多语问答等。
2. Scientific text:其有助于增强LLMs在科学领域的表现以及推理能力。现有的一些语料库通常从arXiv、科学书籍等相关领域上收集数据。但值得注意的是，这些数据通常有各自领域的表达格式，在使用时需要将其转换为模型能够处理的格式。
3. Code:其有助于增强LLMs的推理能力。现有数据主要从编程问答社区如Stack Exchange和GitHub上进行收集。

![](../assets/img/transformers-note-11/pretraining_data.png#id=QToT9&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

经过研究证明，不同种类数据的混合使用有助于增强LLMs的泛化性能；不同种类数据所用的比例影响着LLMs在不同领域的效果；所用预训练数据是否满足LLMs的需求对最终的效果有很大影响。

### 数据预处理

预训练过程中所用语料的质量对所得LLMs最终的效果有着显著的影响，因而数据的预处理显得十分重要，其主要过程如下图所示。

![](../assets/img/transformers-note-11/pretraining_process.png#id=RDFGB&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

本小节对其中的各个过程进行介绍。

- 质量过滤

所用方法主要可分为两类：

1.  classifier-based approaches:通过预先训练一个分类器来对文本的质量进行评测（如二分类下直接将文本分为高质量与低质量两类）。但一些研究发现，该方法可能会导致方言、口语类数据的减少，最终导致在整体语料中出现偏差以及多样性的降低。 
2.  heuristic-based approaches:通过预先设计一些规则来消除低质量文本。比如根据具体指标文本的困惑度（Perplexity）来测定文本的质量、对一些关键词进行过滤等。采用该方法的有BLOOM，Gopher等。 

- 去重

研究发现文本中的重复数据可能会导致训练过程的不稳定，从而影响模型的效果，所以需要对预训练语料进行去重。可从三个层面上进行：

1. Sentence-level:直接去除那些包含重复单词或者短语的句子。
2. Document-level:现有一些去重方法主要是通过测量不同文档间一些特征（如单词）的重复率来进行去重。
3. Dataset-level:如避免训练集和验证集中数据的重复。

- 隐私保护

为了避免所用语料中出现个人隐私泄露的问题，需要对其中涉及到的个人信息进行消除，通常使用rule-based methods如直接对name、addresses等信息进行过滤。

- 分词

Tokenization将原始文本转化为token序列从而作为LLMs的输入。尽管使用现有一些分词方法如BPE、Wordpiece等能够取得一定的效果，但也有很多LLMs根据自身所用语料定制化训练分词器（tokenizer）以适配所用数据，但在后续使用时要注意训练分词器所用语料的语言与特定任务中数据语言存在差异时所带来的影响。

### 主流模型结构

无可质疑的是，Transformer仍是LLMs的backbone，但依据结构上的差异，可以将现有的主流结构分为三类：

1.  Encoder-decoder Architecture:
即原始Transfomer中使用的架构，具体内容可参考第三章的介绍。迄今为止，使用该结构的LLMs数量很少，具体有T5,BART等。 
2.  Casual Decoder Architecture:
与下面介绍的结构可被共同称为Decoder-only architectures。GPT系列模型就采用此种结构，也是目前大部分LLMs所用的结构如OPT,BLOOM,Gopher。它使用单向注意力掩码机制，从而保证输入中的token只能看到它本身和位置位于它之前的token。 
3.  Prefix Decoder Architectures:
该结构在输入部分采用双向注意力，即输入间可以相互看到；但在输出时仍采用Casual Decoder中的单向注意力。在使用上，通常的做法是以现有的casual decoders为基础继续训练，比如U-PaLM就源于PaLM。 

![](../assets/img/transformers-note-11/pretraining_model.png#id=CGjzQ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

在图中我们可以通过蓝色部分很清晰的看到casual decoder和prefix decoder的不同。

在Transformer发布后，后续提出了很多方面的改进以提升模型训练的稳定性和最终效果，大体有Normalization position、Normalization method、Activation function、Position embedding等方面，涉及的细节部分这里不再介绍，在下图中针对这些内容有一个概括性的陈列。

![](../assets/img/transformers-note-11/pretraining_detail.png#id=xWXHW&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

为了获得更强的泛化和训练稳定性，在实施中建议选择 RMSNorm 进行layer normalization，将 SwiGLU 或 GeGLU 作为激活函数。此外，layer normalization 如果在嵌入层之后立即使用可能会导致性能下降。在position embedding方面，RoPE 或 ALiBi由于其在长序列上的更优表现可能是更好的选择。

由于注意力机制是上述所有基于Transformer结构的LLMs的核心，所以针对这一机制也出现了很多变种。如：

1.  Sparse attention:
由于full attention带来的是n2级别的计算量，所以为了减少计算量，在该种机制下，每个查询只和基于位置得到的子token序列中的token计算注意力。 
2.  Multi-query attention:
它指的是在多头注意力实现中不同的heads共享相同的线性变换矩阵。代表性的LLMs有PaLM和StarCoder。 
3.  FlashAttention:
与上述调整不同，该机制从IO-aware方面直接优化GPUs上注意力模块的执行速度和内存占用，其更好的利用了快速内存SRAM。其已经在Pytorch、DeepSpeed上得到了部署。 

### 预训练任务

预训练过程中通常使用以下三个任务作为训练目标：

1.  Language Modeling:
该任务被广泛地应用在使用Decoder-only的LLMs的预训练中，如GPT3,PaLM。其目标是给定输入序列![](https://www.yuque.com/api/services/graph/generate_redirect/latex?x%3D%7Bx_1%2Cx_2%E2%80%A6%E2%80%A6x_n%7D#card=math&code=x%3D%7Bx_1%2Cx_2%E2%80%A6%E2%80%A6x_n%7D&id=DdBdK),基于![](https://www.yuque.com/api/services/graph/generate_redirect/latex?x_%7B%3Ci%7D#card=math&code=x_%7B%3Ci%7D&id=Rw1xt)这样一组序列以autoregressive的方式预测![](https://www.yuque.com/api/services/graph/generate_redirect/latex?x_i#card=math&code=x_i&id=z20gt)。
![](../assets/img/transformers-note-11/pretraining_LM.png#id=JGBTF&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
因为大多数任务都可以根据输入转换为预测问题，所以以该预训练目标得到的LLMs有利于其以一种统一的方式解决问题。以该目标基础的一个变种任务是prefix language modeling,对应前面提到的prefix decoder architectures的LLMs，在这样的方式下计算损失时，不会考虑prefix部分。 
2.  Denoising Autoencoding:
Denoising autoencoding task(DAE)将输入x中的![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Ctilde%7Bx%7D#card=math&code=%5Ctilde%7Bx%7D&id=FIryT)随机替换，其目标就是恢复被替换的内容![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Ctilde%7Bx%7D#card=math&code=%5Ctilde%7Bx%7D&id=LV63F)。但在实际应用中，由于该任务相对LM task更为复杂，作为预训练目标没有被广泛地使用。现有的LLMs中T5,GLM-130B以DAE作为训练目标。
![](../assets/img/transformers-note-11/pretraining_DAE.png#id=z2O9p&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
3.  Mixture-of-Denoisers：
Mixture-of-Denoisers（MoD）也被称之为UL2损失，是一种用于预训练的统一目标，它将LM和 DAE这两种任务的目标视为不同类型的去噪任务，即S-denoiser（LM）、R-denoiser（DAE，短跨度和低程度的噪声）和X-denoiser（DAE，长跨度或高程度的噪声）。MoD在PaLM2中得到了应用。 

模型结构和预训练目标的不同选择会带来不同的归纳偏差（inductive bias）,从而影响LLMs最终的效果。本节介绍有关方面的两个开放性讨论。

1.  在模型结构选择方面，casual decoder architecture仍是主流选择，但至今没有足够有力的理论以给出证明。有一些观点认为，这种结构能够在zero-shot和few-shot上实现比较好的效果，研究证明在不经过多任务微调的情况下，以这种结构预训练得到的LLMs相较其他结构有更好的zero-shot效果。此外，在下文中介绍的instruction tuing和alignment tuing已经被证明能够进一步增强使用casual decoder architecture的LLMs的能力；同时scaling law在casual decoder models上得到了证实，但与此同时将encoder-decoder model扩展到大规模上进行的研究还不足。 
2.  在长文本方面，由于Transformer本身计算上带来的限制，导致模型可处理的文本长度受到很大的限制。现在GPT-4的上下文窗口（context window）能够处理32k个token。针对这一问题，有两个重要的因素需要考虑。
一个是Extrapolation capability，指的是模型在处理超出其训练数据范围之外的输入时的能力。它衡量了模型是否能够在未见过的情况下进行准确的预测或生成，体现着模型的泛化能力。
另一个是Efficiency，即进一步提升计算效率，如前文提到的sparse attention、FlashAttention等。 

### 优化设置

在优化方面，我们简单介绍以下几个方面：

1.  Batch Training:
现有的一些实验结果表明batch size的动态调整有助于增强LLMs训练过程的稳定，例如在GPT-3中batch size从32K逐步增加到3.2M。 
2.  Learning Rate：
现有的LLMs在训练中通常采用warm-up和decay strategies。具体来说，在初始 0.1% 到 0.5% 的训练步骤中，采用一种linear warm-up的方式将学习率逐渐增加到最大值，范围从大约 5 × 10−5 到 1 × 10−4（如GPT-3 是 6 × 10−5）。然后，后续步骤中采用余弦衰减策略，将学习率逐渐降低到其最大值的大约 10%，直到训练损失的收敛。 
3.  Optimizer:
通常采用Adam和AdamW作为optimizer（如GPT-3），参数设置：![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbeta_1#card=math&code=%5Cbeta_1&id=JU9p2)=0.9，![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbeta_2#card=math&code=%5Cbeta_2&id=E4Xxb)=0.95,![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cepsilon#card=math&code=%5Cepsilon&id=Lnrbx)=![](https://www.yuque.com/api/services/graph/generate_redirect/latex?10%5E%7B-8%7D#card=math&code=10%5E%7B-8%7D&id=vEiNx)；也有一些采用Adafactor作为optimizier（如PaLM和T5），参数设置：![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbeta_1#card=math&code=%5Cbeta_1&id=u0ZCH)=0.9，![](https://www.yuque.com/api/services/graph/generate_redirect/latex?%5Cbeta_2#card=math&code=%5Cbeta_2&id=kSSKu)=![](https://www.yuque.com/api/services/graph/generate_redirect/latex?1.0-k%5E%7B-0.8%7D#card=math&code=1.0-k%5E%7B-0.8%7D&id=r0sfP),k为training steps的数目。 
4.  Stabilizing the Training：
为了尽可能避免训练过程中出现instability issue，通常采用weight decay(rate一般为0.1)和gradient clipping(threshold一般为1.0)的方式。但随着模型规模的加大，训练损失会出现波动，所以一些LLMs如PaLM和OPT在问题出现后从最近的checkpoint重新开始训练并跳过那些可能造成问题的数据。 

![](../assets/img/transformers-note-11/pretraining_opt.png#id=jH6Qe&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

### 可扩展训练技术

随着模型规模和数据量的加大，如何在已有计算资源的前提下尽量更有效率地完成预训练过程显得十分重要。本节介绍一些常用的方法。

1.  3D Parallelism：
是Data parallelism，Pipline parallelism，Tensor parallelism的统称。
Data parallelism在多个 GPU 上复制模型参数和优化器状态，然后将整个训练语料库分发给这些 GPU。这样，每个 GPU 只需要处理为其分配的数据，并执行前向和后向传播以获得梯度，然后进一步聚合不同 GPU 上计算的梯度以获得整个批次的梯度，以更新所有 GPU 中的模型。通过这种方式，能够通过增加 GPU 的数量以提高训练吞吐量。
Pipline parallelism旨在将LLM的不同层分布到多个GPU中。特别是，在使用 Transformer 模型的情况下，它将连续的层加载到同一个 GPU 上，以降低在 GPU 之间传输计算的隐藏状态或梯度的成本。但一个可能的问题是，这样会导致较低的GPU利用率，因为每个GPU必须等待前一个GPU完成计算。为了缓解这一问题，GPipe和PipeDream提出填充多批数据和异步梯度更新的技术。
与Pipline parallelism不同，Tensor parallelism侧重于分解 LLM 中的Tensor（即参数矩阵）。对于LLM中的矩阵乘法操作Y = XA，参数矩阵A可以按列分成两个子矩阵A1和A2，进而表示为Y = [XA1, XA2]。通过在不同的 GPU 上放置矩阵 A1 和 A2，矩阵乘法操作将在两个 GPU 上并行调用，并且可以通过跨 GPU 通信组合两个 GPU 的输出来获得最终结果。
在具体使用中，通常将这三种方式结合使用。 
2.  ZeRO:
该技术由Deip-Speed提出，为了解决Data parallelism中存在的内存冗余问题。如前所述，Data parallelism要求每个 GPU 存储 LLM 状态的副本，包括模型参数、模型梯度和优化器参数。然而，并非所有上述数据都需要在每个 GPU 上保留。ZeRO 技术旨在每个 GPU 上只保留一小部分数据，而在需要时可以从其他 GPU 中检索其余数据。具体来说，ZeRO 提供了三个解决方案，具体取决于数据的三部分如何存储，即optimizer state partitioning, gradient partitioning, and parameter partitioning。实验表明前面两部分并未增加通信开销，而第三部分虽然增加了大约50%的通信开销但同时节省了与GPU数量成比例的内存。 
3.  Mixed Precision Training:
在早期通常使用FP32的数据格式，后来开始使用FP16格式以减少内存使用和通信开销，但一些研究发现FP16可能会导致精度上的损失从而影响模型的最终效果。为了缓解该问题，后续有研究者使用Brain Floating Point(BF16)用于训练。 

由于LLM的训练是一个时间成本和资源成本都耗费巨大的过程，所以如何在早期阶段对模型未来的效果进行预测以及检测训练过程中的问题变得很重要（比如GPT-4引入了一种新的机制，称为建立在深度学习堆栈上的predictable scaling，从而能够使用更小的模型预测LLM的效果）。

[toc]

## 三、大语言模型的适应性

	经过预训练，LLMs可以获得解决各种任务的一般能力。然而，越来越多的研究表明，LLMs的能力可以根据具体目标进一步调整。在本节中，我们将介绍调整预训练LLMs的两种主要方法，即`instruction tuning`和`alignment tuning`。前一种方法旨在增强(或解锁)LLMs的能力，而后一种方法旨在使LLMs的行为与人类的价值观或偏好保持一致。此外，我们还将讨论资源有限环境下模型适应的有效调优和量化。接下来，我们将详细介绍这四个部分。

### 1. Instruction Tuning

	本质上，instruction tuning是在自然语言形式的格式化实例集合上微调预先训练的LLMs的方法，这与监督微调和多任务提示训练高度相关。为了执行instruction tuning，我们首先需要收集或构造instruction-formatted的实例。然后，我们使用这些格式化的实例以监督学习的方式微调 LLMs（例如，使用序列到序列损失进行训练）。在instruction tuning之后，LLMs可以表现出泛化到未知任务的能力，即使在多语言环境中。

1.  格式化实例构造
通常，一个instruction-formatted实例由任务描述（称为instruction）、可选输入、相应输出和少量演示（可选）组成。接下来，我们介绍了构建格式化实例的三种主要方法，然后讨论了实例构建的几个关键因素。
![](../assets/img/transformers-note-11/Adaptation_Figure_1.png#id=oFv2y&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 图1. 实例格式的说明和构造指令格式实例的三种不同方法 
   -  格式化任务数据集
	使用自然语言任务描述对这些多任务训练数据集进行格式化很方便。具体来说，最近的工作用人工编写的任务描述增强了标记数据集，从而指导 LLMs 通过解释任务目标来理解任务。例如，在问答任务中每个示例中都添加了一个任务描述“请回答这个问题”。通过在删除任务描述的标记数据集上微调模型，它会导致模型性能急剧下降，这表明instructions是 LLMs 的任务泛化能力的关键因素。为了丰富训练实例，一些研究也尝试使用专门设计的任务描述来反转现有实例的输入输出对以进行instruction tuning。例如，给定一个问答对，我们可以通过预测问题条件答案来创建一个新实例（例如，“请根据答案生成问题：”）。 
   -  格式化每日聊天数据
	尽管大量的训练实例已经用instruction进行了格式化，但它们主要来自公共的NLP数据集，要么缺乏instruction多样性，要么与人类的真实需求不匹配。为了克服这个问题，InstructGPT建议采用真实用户提交到OpenAI API的查询作为任务描述。为了丰富任务的多样性，还要求人工标注者为现实生活中的任务编写instructions，包括开放式生成、开放式问答、头脑风暴和聊天。然后，他们让另一组标注者直接回答这些instructions作为输出。最后，他们将一条instruction(即收集到的用户查询)和预期输出(即人工编写的答案)配对作为训练实例。 
   -  格式化合成数据
	为了减轻人工注释或手动收集的负担，已经提出了几种半自动化方法，通过将现有实例馈送到LLMs中来构建实例，以合成各种任务描述和实例。 
   -  实例构建的关键因素 
      -  缩放instructions。大量研究表明，扩展任务数量可以极大地增强LLMs的泛化能力。 
      -  格式设计。
作为一个重要因素，自然语言格式的设计也高度影响着LLMs的泛化性能。通常，我们可以在现有数据集的输入输出对中添加任务描述和可选演示，其中任务描述是LLMs理解任务的最关键部分。此外，通过使用适当数量的范例作为演示，它可以带来实质性的改进，这也减轻了模型对instruction工程的敏感性。
最近，为了引出LLMs的分步推理能力，一些研究提出在一些推理数据集(如算术推理)中加入思维链(CoT)示例。研究表明，对带有CoT和非CoT示例的LLMs进行微调可以在各种推理任务中获得良好的性能，包括那些需要多跳推理能力的任务。
总之，instructions的多样性和质量似乎比实例的数量更重要，因为性能良好的InstructGPT和Alpaca比flan系列LLMs使用更少但更多样化的instructions(或实例)。此外，邀请标注者来编写人类需要的任务比使用特定于数据集的任务更有用。 
2.  Instruction Tuning 策略
与预训练不同，instruction tuning通常更有效，因为只使用适量的实例进行训练。由于instruction tuning可以被认为是一个有监督的训练过程，因此它的优化与预训练有几个方面的不同，例如训练目标(即序列到序列的损失)和优化配置(例如更小的批大小和学习率)，这些都需要在实践中特别注意。除了这些优化配置之外，还有两个重要的方面需要考虑: 
   - 均衡数据分布。根据最近的研究结果，增加高质量集合(如FLAN和P3)的采样比例通常可以提高性能。
   - instruction tuning与预训练相结合。一些研究没有使用单独的两阶段过程(预训练然后instruction tuning)，而是尝试使用多任务学习，使用预训练数据(即纯文本)和instruction tuning数据(即格式化数据集)的混合从头开始训练模型。
3.  Instruction Tuning的效果 
   - 提高性能。尽管在适度数量的实例上进行了调优，但instruction tuning已经成为提高或解锁LLMs能力的重要方式。最近的研究对多个尺度(77M到540B)的语言模型进行了实验，结果表明，不同尺度的模型都可以从instruction tuning中受益，随着参数尺度的增加，性能得到提高。此外，具有instruction tuning的较小模型甚至可以比没有微调的较大模型表现得更好。除了模型规模之外，instruction tuning在各种模型架构、预训练目标和模型自适应方法上都有一致的改进。在实践中，instruction tuning提供了一种增强现有语言模型(包括小型PLMs)能力的通用方法。此外，它比预训练成本低得多，因为LLMs所需的instruction数据量明显小于预训练数据。
   - 任务泛化。大量的研究已经证实了instruction tuning的有效性，可以在可见任务和不可见任务上获得卓越的性能。此外，instruction tuning已被证明有助于缓解LLMs的几个弱点(例如，重复生成或补充输入而未完成某个任务)，从而使LLMs具有解决实际任务的卓越能力。此外，经过instruction tuning训练的LLMs推广到跨语言的相关任务。
   - 领域专业化。instruction tuning是使现有的通用LLMs成为特定领域专家的有效方法。
4.  Instruction Tuning的实证分析
使用不同instruction集的微调LLMs往往会导致在下游任务上具有不同性能的模型变体。我们总结了在现有工作中广泛使用的四种主要改进策略: 
   - 提高instruction复杂度。正如已有研究所讨论的那样，提高instructions的复杂性可以提高LLMs遵循复杂instructions的模型能力，例如包括更多的任务需求或需要更多的推理步骤。
   - 增加话题的多样性。除了复杂性之外，提高instructions数据集的主题多样性有助于激发LLMs在现实世界中处理不同任务的不同能力。
   - 缩放instruction数。除了以上几个方面，instructions的数量也是可能影响模型性能的一个重要因素。
   - 平衡instruction难度。由于合成instructions往往包含过于简单或过于困难的instruction，这很可能导致LLMs的训练不稳定甚至过拟合。

一些研究实验结果表明： 

   - 任务格式的instruction更适合QA设置，但可能不适用于聊天设置。
   - 多种instructions形式的结合对提高LLMs的综合能力有很大的帮助。
   - 增强instruction的复杂性和多样性可以提高模型的性能。
   - 简单地增加instruction数量可能并不那么有用，平衡难度也并不总是有用的。在没有质量控制的情况下，单纯地增加合成instruction的数量可能无法有效地提高性能。

### 2. Alignment Tuning

	本部分首先介绍了alignment的背景及其定义和标准，然后介绍了用于aligning LLMs的人类反馈数据的收集，最后讨论了用于alignment tuning的人类反馈强化学习(RLHF)的关键技术。

1.  alignment背景和标准
	LLMs在广泛的NLP任务中显示出卓越的能力。然而，这些模型有时可能会表现出意想不到的行为，例如，编造虚假信息，追求不准确的目标，并产生有害的、误导性的和有偏见的表达。对于LLMs，语言建模目标通过单词预测预训练模型参数，而不考虑人的价值观或偏好。为了避免这些意想不到的行为，人们提出了human alignment，使LLMs的行为符合人类的期望。
	近年来，人们越来越关注制定各种标准来规范LLMs的行为。三个代表性的对齐标准：有益的、诚实的和无害的。这些标准在现有文献中被广泛采用。此外，LLMs还从行为、意图、激励和内在等不同角度提出了其他的对齐标准，这些标准与上述三个标准本质上相似(或至少具有相似的对齐技术)。 
2.  收集人们的反馈
	在预训练阶段，LLMs使用大规模语料库上的语言建模目标进行训练。然而，它不能考虑到人类对LLMs产出的主观和定性评价(在本文中称为人类反馈)。高质量的人类反馈对于使LLMs与人类的偏好和价值观保持一致非常重要。在现有的工作中，主要有三种方法来收集人类标注者的反馈和偏好数据。 
   1. 在早期的工作中，人类标注者通常以粗粒度的方式评估模型生成的输出(即只选择最好的)，而不考虑更细粒度的对齐标准。尽管如此，不同的标注者可能对最佳候选输出的选择持有不同的意见，并且该方法忽略了未选择的样本，这可能导致不准确或不完整的人为反馈。
   2. 人类标注者可以通过回答研究人员设计的某些问题来提供更详细的反馈，这些问题涵盖了对齐标准以及LLMs的附加约束。特别是，在WebGPT中，为了帮助模型从检索到的文档中过滤和利用相关信息，需要人工标注者回答关于检索到的文档是否对回答给定输入有用的多个选项的问题。
   3. 许多研究还开发了基于规则的方法来提供更详细的人类反馈。作为一个典型案例，Sparrow不仅选择了标注者认为最好的响应，而且还使用一系列规则来测试模型生成的响应是否满足有益、正确和无害的对齐标准。
3.  基于人类反馈的强化学习
	为了使LLMs与人类价值观保持一致，已经提出了基于人类反馈的强化学习(RLHF)，利用收集到的人类反馈数据对LLMs进行微调，这有助于提高对齐标准(例如，有用性、诚实性和无害性)。RLHF采用强化学习(RL)算法(例如，Proximal Policy Optimization (PPO))通过学习奖励模型使LLMs适应人类反馈。这种方法将人类纳入训练循环中，以开发对齐良好的LLMs，如InstructGPT所示。
	RLHF系统主要由三个关键部分组成:一个预训练的LM，一个从人类反馈中学习的奖励模型，以及一个训练LM的RL算法。具体来说，预训练的LM通常是一个生成模型，使用现有的预训练的LM参数进行初始化。例如，OpenAI使用175B GPT-3作为其第一个流行的RLHF模型InstructGPT， DeepMind使用2800亿个参数模型Gopher作为其GopherCite模型。此外，奖励模型(RM)提供(学习到的)指导信号，这些信号反映了人类对LM生成的文本的偏好，通常以标量值的形式出现。奖励模型可以采取两种形式:微调的LM或使用人类偏好数据重新训练的LM。现有工作通常采用的奖励模型的参数尺度与对齐的LM不同。例如，OpenAI使用6B GPT-3, DeepMind使用7B Gopher作为奖励模型。最后，为了使用来自奖励模型的信号来优化预训练的LM，设计了一种用于大规模模型调谐的特定强化学习算法。具体来说，近端策略优化(Proximal Policy Optimization, PPO)是一种在现有工作中广泛使用的RL对齐算法。
![](../assets/img/transformers-note-11/Adaptation_Figure_2.png#id=XBykb&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 图2. RLHF算法的工作流程
RLHF的关键步骤： 
   1. 监督微调。为了使LM最初执行所需的行为，它通常需要收集一个受监督的数据集，其中包含输入提示(Instruction )和用于微调LM的所需输出。
   2. 奖励模式训练。第二步是使用人类反馈数据来训练RM。具体来说，我们使用LM来生成一定数量的输出文本，使用采样提示(来自监督数据集或人工生成的提示)作为输入。然后，我们邀请人类标记员来注释这些对的偏好。注释过程可以有多种形式，常用的方法是通过对生成的候选文本进行排序来进行注释，这样可以减少注释者之间的不一致性。然后，训练RM来预测人类偏好的输出。在InstructGPT中，标签器将模型生成的输出从最好到最差排序，RM(即6B GPT-3)被训练来预测排名。
   3. RL微调。在这个步骤中，对齐(即微调)LM被形式化为RL问题。由于RL算法的不稳定性，最近的研究通过重用具有更高奖励的最佳排名样本，用另一种监督微调取代了RL调优。

### 3. 参数高效的模型适应

	在上面，我们讨论了根据特定目标调整LLMs的instruction tuning和alignment tuning方法。由于LLMs由大量的模型参数组成，因此执行全参数调优的成本很高。在本节中，我们将讨论如何对LLMs进行有效的调优。在现有文献中，参数高效微调一直是一个重要的主题，旨在减少可训练参数的数量，同时尽可能保持良好的性能。我们简要回顾Transformer语言模型的四种参数高效的微调方法，包括适配器调优、前缀调优、提示调优和LoRA。

![](../assets/img/transformers-note-11/Adaptation_Figure_3.png#id=VpnNJ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

1. 适配器调优将小型神经网络模块(称为适配器)集成到Transformer模型中。适配器模块将集成到每个Transformer层中，通常在Transformer层的两个核心部分(即注意层和前馈层)之后使用串行插入。在微调过程中，适配器模块将根据特定的任务目标进行优化，而原始语言模型的参数在此过程中被冻结。这样，我们可以在微调过程中有效地减少可训练参数的数量。
2. 前缀调优为语言模型中的每个Transformer层添加一系列前缀，这些前缀是一组可训练的连续向量。这些前缀向量是特定于任务的，可以将其视为虚拟令牌嵌入。由于只训练前缀参数，因此可以实现参数有效的模型优化。
3. 与前缀调优不同，提示调优主要侧重于在输入层加入可训练的提示向量。基于离散提示方法，它通过包括一组软提示令牌(自由形式或前缀形式)来增强输入文本，然后使用增强的提示输入来解决特定的下游任务。在训练过程中，根据特定任务的监督，只学习提示嵌入。
4. LoRA在每个密集层上施加低秩约束来逼近更新矩阵，从而减少可训练参数以适应下游任务。

	LoRA已被广泛应用于开源llm(例如LLaMA和BLOOM)，用于参数高效的微调。在这些研究尝试中，LLaMA及其变体在参数高效调谐方面受到了广泛关注。例如，AlpacaLoRA已经使用LoRA作为Alpaca的轻量级调优版本进行了训练(一种经过微调的7B LLaMA模型，具有52K的人类指令演示)。有研究比较了四种有效的调优方法，包括串行适配器调优，并行适配器调优和LoRA，LoRA在这些比较方法中表现相对较好，使用的可训练参数明显较少。

	作为一个重要的资源，库PEFT(代表参数高效微调)已经在GitHub32上发布。它包括几种广泛使用的高效调优方法，包括LoRA /AdaLoRA、prefixtuning、P-Tuning和prompt-tuning。此外，它支持许多语言模型，例如GPT-2和LLaMA，并且还涵盖了几个代表性的视觉Transformer模型(例如，ViT和Swin Transformer)。

### 4. 内存高效的模型适应

	由于有大量的模型参数，LLMs需要占用大量的内存来进行推理，这使得在实际应用程序中部署LLMs的成本非常高。在本节中，我们将讨论如何通过流行的模型压缩方法(即模型量化)减少LLMs的内存占用，以便大型LLMs可以在资源有限的环境中使用，这也可能减少推理延迟。

	通常有两种主要的模型量化方法，即`量化感知训练(QAT)`(需要额外的全模型再训练)和`训练后量化(PTQ)`(不需要模型再训练)。与小型语言模型相比，在设计或选择LLMs量化方法时需要考虑两个主要差异。首先，LLMs由大量参数组成，因此PTQ方法比QAT方法计算成本低得多，因此更受欢迎。其次，LLMs表现出非常不同的激活模式(即大的离群值特征)，并且LLMs的量化变得更加困难，特别是隐藏激活。

一些LLMs的PTQ方法：

- Mixed-precision分解（Mixed-precision decomposition）。
- 细粒度的量子化（Fine-grained quantization）。
- 平衡量化难度（Balancing the quantization difficulty）。
- Layerwise量子化（Layerwise quantization）。

其他量化方法：

- 高效微调增强量化（Efficient fine-tuning enhanced quantization）。
- LLMs量化意识培训(QAT)（Quantization-aware training (QAT) for LLMs）。

## 四、大模型的使用

![](/assets/img/transformers-note-11/utilization_ICL_vs_CoT.png#id=Q5ICC&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

### In-Context Learning

ICL的形式如上图，由三个部分组成：任务描述，若干个问答的示例（demonstration）和一个query

ICL与前文提及的指令微调，都采用了自然语言的形式来描述任务和实例，区别在于后者是为了适配模型，前者仅在使用阶段提示LLM

ICL的性能很大程度上依赖于示例，所以设计合适的示例至关重要，这包含以下三个方面：

-  示例选择：通常采用简单且低开销的启发式方法，比如基于k-NN检索与query语义相关的示例、选择最具代表性的示例集合 
-  示例格式：一些研究考虑添加任务描述，或者通过 CoT 提示来构建模板 
-  示例顺序：LLM会倾向于重复示范结尾的答案 。早期工作提出了一些启发式方法来快速地找到一个良好的顺序 

关于LLM如何实现ICL：

-  一种观点认为，LLM主要是从示例中识别出目标任务而非从中学习。也就是说，LLM从示例中识别出目标任务后，利用预训练中获得的先验知识来解决新任务。相关研究认为，在预训练数据中存在一个代表任务的潜在变量，LLM能够从示例中捕获这个变量，使它们能够在ICL中识别任务。 
-  另一种观点认为，LLM仅仅通过示例来学习在预训练阶段没有见过的新任务。因为任务识别的能力在小型LM中也能体现，而只有LLM才能涌现任务学习的能力。同时一些研究表明，小型LM倾向于忽略标签，主要依靠其先验知识来完成任务，而LLM有能力超越其先验知识，并从示例中获取新知识，从而获得更好的结果。 

### Chain-of-Thought Prompting

CoT是一种改进的Prompting，主要用于复杂推理任务中。我们通常在ICL中使用CoT。

- Few-shot CoT：是ICL的一种特例，即CoT在输入输出之间额外包含了中间推理步骤。一种直接的方法是使用多样的推理路径，通过这些答案得到最一致的答案。另一个基于直觉的想法是，具有复杂推理路径的提示更有可能引出 LLM 的推理能力，这可以提高生成正确答案的准确性。
- Zero-shot CoT：即，在提示中不包含人类标注的示例。比如，简单地使用“Let’s think step by step”提示 LLM 来生成推理步骤，然后通过“Therefore, the answer is”来得出最终答案。

CoT何时适用于LLM?
CoT是大模型中的一种涌现能力，通常用于需要逐步推理的任务，而在一些简单任务中反而没有标准提示的效果好。

LLM为何能有CoT?
由于在代码数据中训练的模型具有强大的推理能力，所以通常认为CoT的能力也源于此，但这并未得到充分实验证明。另外，有实验表明，在非 CoT 数据上进行指令微调不会提高模型使用 CoT 完成任务的性能，所以这也不是CoT的关键因素。
有研究表明，模式（比如算术推理的公式）和文本较为重要，而符号（比如算术推理的数值量）和模式的正确性却不重要，并且文本和模式是相互促进的。

### Planning for Complex Task Solving

ICL和CoT都是在各种任务上比较通用的方法，缺陷在于难以解决数学推理这种复杂任务，为此提出基于提示的Planning方法，将一个复杂任务拆分为多个子任务，并生成完成任务的行动计划，逐一解决这些任务。

整体框架如下图，由LLM理解目标任务后生成计划（既可以是自然语言的行动序列，也可以是编程语言的可执行程序），Executor（LLM或机器人）在环境中执行计划，环境把关于行动结果的反馈（自然语言或其他模态的信号）返回LLM，从而让LLM改善计划，不断重复此过程以得到更好的结果。

![](/assets/img/transformers-note-11/utilization_planning_framework.png#id=JxOKV&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

### Prompt设计指南

Prompting是使用LLM的主要方法，而prompts的质量在很大程度上影响LLM在特定任务的性能，接下来让我们看看如何设计合适的Prompts

Prompts包含了四个关键要素：

- task description：用自然语言的形式描述任务目标，我们需要进一步使用关键词强调特殊设置（比如特别的输入输出格式）来引导LLM
- input data：通常我们可以直接以自然语言的形式描述输入数据，而对于知识图和表格等结构化数据，我们通常线性化为序列。另外，编程语言的可执行代码也可用于形式化这些数据
- contextual information：例如，检索到的文档对于开放域问答非常有用，可以作为支持证据。因此，它需要以适当的提示模式或表达式格式包含这些信息。此外，上下文任务示例也有助于激发llm完成复杂任务，它可以更好地描述任务目标、特殊的输出格式以及输入和输出之间的映射关系。
- prompt style：prompt应该表达为清晰的问题和详细的指令。有时添加前缀和后缀可以更好地引导LLM，比如添加前缀“Let us think step by step”或“You are an expert on this task（or in this domain）”。另外，对于chat-based的LLM（比如ChatGPT），比起直接输入冗长且复杂的prompt，把它分解为与子任务对应的多个prompt，以多轮对话的形式输入LLM，可能有更好的效果。

一些设计原则如下：

- 我们需要清楚地表达任务目标，如Given a long document, I want you to generate a concise summary，并说清楚条件限制，如the length of the summary cannot exceed 50
- 尽量划分为简单的子任务，如Braid a coherent narrative by performing the following tasks: 1. ...; 2. ...; 3. ...
- 提供少量高质量的示例
- 使用模型友好的格式：在OpenAI文档中建议使用###或"""分离指令和上下文，可以更好地被LLM理解
- 复杂任务中有特定的输出形式或背景知识，prompt显得更为重要，能达到与监督方法相当或更好的性能。如数学推理任务，可以基于编程语言的形式设计prompt

## 参考

[[1]](https://github.com/RUCAIBox/LLMSurvey) 大语言模型综述
