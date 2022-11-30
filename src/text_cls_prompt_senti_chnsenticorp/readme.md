### 文本分类任务（Prompt情感分析）

详细说明请见[《Transformers 库快速入门 第十三章：Prompt 情感分析》](https://transformers.run/nlp/2022-10-10-transformers-note-10.html)

运行 *run_prompt_senti_bert.sh* 脚本即可进行训练。

```
bash run_prompt_senti_bert.sh
```

如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的 Macro-F1 值和 Micro-F1 值都达到  95.33%（积极: 96.62 / 94.08 / 95.33, 消极: 94.08 / 96.62 / 95.33）（Nvidia Tesla V100, batch=4）。
