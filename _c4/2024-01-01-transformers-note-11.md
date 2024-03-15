---
title: 第十四章：大模型简介
author: SHENG XU
date: 2024-01-01
category: NLP
layout: post
---

！！！这行不要删：请保留两级目录，使用（##）和（###）开头。







## 大模型的使用

<img src="/assets/img/transformers-note-11/utilization_ICL_vs_CoT.png" alt="utilization_ICL_vs_CoT" style="display: block; margin: auto; width: 800px">



### In-Context Learning

ICL的形式如上图，由三个部分组成：任务描述，若干个问答的示例（demonstration）和一个query

ICL与前文提及的指令微调，都采用了自然语言的形式来描述任务和实例，区别在于后者是为了适配模型，前者仅在使用阶段提示LLM

ICL的性能很大程度上依赖于示例，所以设计合适的示例至关重要，这包含以下三个方面：
- 示例选择：通常采用简单且低开销的启发式方法，比如基于k-NN检索与query语义相关的示例、选择最具代表性的示例集合

- 示例格式：一些研究考虑添加任务描述，或者通过 CoT 提示来构建模板

- 示例顺序：LLM会倾向于重复示范结尾的答案 。早期工作提出了一些启发式方法来快速地找到一个良好的顺序


关于LLM如何实现ICL：
- 一种观点认为，LLM主要是从示例中识别出目标任务而非从中学习。也就是说，LLM从示例中识别出目标任务后，利用预训练中获得的先验知识来解决新任务。相关研究认为，在预训练数据中存在一个代表任务的潜在变量，LLM能够从示例中捕获这个变量，使它们能够在ICL中识别任务。

- 另一种观点认为，LLM仅仅通过示例来学习在预训练阶段没有见过的新任务。因为任务识别的能力在小型LM中也能体现，而只有LLM才能涌现任务学习的能力。同时一些研究表明，小型LM倾向于忽略标签，主要依靠其先验知识来完成任务，而LLM有能力超越其先验知识，并从示例中获取新知识，从而获得更好的结果。

  


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

<img src="/assets/img/transformers-note-11/utilization_planning_framework.png" alt="utilization_planning_framework" style="display: block; margin: auto; width: 800px">



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
