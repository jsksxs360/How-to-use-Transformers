![title](title.jpg)

[Transformers](https://huggingface.co/docs/transformers/index) 是由 [Hugging Face](https://huggingface.co/) 开发的一个 NLP 包，支持加载目前绝大部分的预训练模型。随着 BERT、GPT 等大规模语言模型的兴起，越来越多的公司和研究者采用 Transformers 库来构建 NLP 应用。

该项目为[《Transformers 库快速入门》](https://transformers.run/)系列教程的代码仓库，按照以下方式组织代码：

- *data*：存储使用到的数据集；
- *src*：存储所有的任务 Demo，每个任务一个文件夹，可以下载下来单独使用。

## Transformers 库快速入门

- **第一部分：背景知识**
  - 第一章：[自然语言处理](https://transformers.run/back/nlp/)

  - 第二章：[Transformer 模型](https://transformers.run/back/transformer/)

  - 第三章：[注意力机制](https://transformers.run/back/attention/)

- **第二部分：初识 Transformers**
  - 第四章：[开箱即用的 pipelines](https://transformers.run/intro/2021-12-08-transformers-note-1/)
  - 第五章：[模型与分词器](https://transformers.run/intro/2021-12-11-transformers-note-2/)
  - 第六章：[必要的 Pytorch 知识](https://transformers.run/intro/2021-12-14-transformers-note-3/)
  - 第七章：[微调预训练模型](https://transformers.run/intro/2021-12-17-transformers-note-4/)

- **第三部分：Transformers 实战**
  - 第八章：[快速分词器](https://transformers.run/nlp/2022-03-08-transformers-note-5.html)
  - 第九章：[序列标注任务](https://transformers.run/nlp/2022-03-18-transformers-note-6.html)
  - 第十章：[翻译任务](https://transformers.run/nlp/2022-03-24-transformers-note-7.html)
  - 第十一章：[文本摘要任务](https://transformers.run/nlp/2022-03-29-transformers-note-8.html)
  - 第十二章：[抽取式问答](https://transformers.run/nlp/2022-04-02-transformers-note-9.html)
  - 第十三章：[Prompt 情感分析](https://transformers.run/nlp/2022-10-10-transformers-note-10.html)

## Demo 一览

- [pairwise_cls_similarity_afqmc](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/pairwise_cls_similarity_afqmc)：句子对分类任务，[金融同义句判断](https://xiaosheng.run/2021/12/17/transformers-note-4.html)。
- [sequence_labeling_ner_cpd](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/sequence_labeling_ner_cpd)：序列标注任务，[命名实体识别](https://xiaosheng.run/2022/03/18/transformers-note-6.html)。
- [seq2seq_translation](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/seq2seq_translation)：seq2seq任务，[中英翻译](https://xiaosheng.run/2022/03/24/transformers-note-7.html)。
- [seq2seq_summarization](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/seq2seq_summarization)：seq2seq任务，[文本摘要](https://xiaosheng.run/2022/03/29/transformers-note-8.html)。
- [sequence_labeling_extractiveQA_cmrc](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/sequence_labeling_extractiveQA_cmrc)：序列标注任务，[抽取式问答](https://xiaosheng.run/2022/04/02/transformers-note-9.html)。
- [text_cls_prompt_senti_chnsenticorp](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/text_cls_prompt_senti_chnsenticorp)：文本分类任务，[Prompt 情感分析](https://xiaosheng.run/2022/10/10/transformers-note-10.html)。
