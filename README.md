![title](title.jpg)

[Transformers](https://huggingface.co/docs/transformers/index) 是由 [Hugging Face](https://huggingface.co/) 开发的一个 NLP 包，支持加载目前绝大部分的预训练模型。随着 BERT、GPT 等大规模语言模型的兴起，越来越多的公司和研究者采用 Transformers 库来构建 NLP 应用。

该项目为[《Transformers 库快速入门》](https://xiaosheng.run/2021/12/08/transformers-note-1.html)系列教程的代码仓库，按照以下方式组织代码：

- *data*：存储使用到的数据集；
- *src*：存储所有的任务 Demo，每个任务一个文件夹，可以下载下来单独使用。

## Transformers 库快速入门

- **第一部分：初识 Transformers**

  - 第一章：[开箱即用的 pipelines](https://xiaosheng.run/2021/12/08/transformers-note-1.html)
  - 第二章：[模型与分词器](https://xiaosheng.run/2021/12/11/transformers-note-2.html)
  - 第三章：[必要的 Pytorch 知识](https://xiaosheng.run/2021/12/14/transformers-note-3.html)
  - 第四章：[微调预训练模型](https://xiaosheng.run/2021/12/17/transformers-note-4.html)

- **第二部分：Transformers 实战**

  - 第五章：[快速分词器](https://xiaosheng.run/2022/03/08/transformers-note-5.html)
  - 第六章：[序列标注任务](https://xiaosheng.run/2022/03/18/transformers-note-6.html)
  - 第七章：[翻译任务](https://xiaosheng.run/2022/03/24/transformers-note-7.html)
  - 第八章：文本摘要任务
  - 第九章：抽取式问答

## Demo 一览

- [pairwise_cls_similarity_afqmc](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/pairwise_cls_similarity_afqmc)：句子对分类任务，[金融同义句判断](https://xiaosheng.run/2021/12/17/transformers-note-4.html)。
- [sequence_labeling_ner_cpd](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/sequence_labeling_ner_cpd)：序列标注任务，[命名实体识别](https://xiaosheng.run/2022/03/18/transformers-note-6.html)。
- [seq2seq_translation](https://github.com/jsksxs360/How-to-use-Transformers/tree/main/src/seq2seq_translation)：seq2seq任务，[中英翻译](https://xiaosheng.run/2022/03/24/transformers-note-7.html)。
