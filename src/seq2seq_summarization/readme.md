### Seq2Seq任务（文本摘要）

详细说明请见[《Transformers 库快速入门 第十一章：文本摘要任务》](https://transformers.run/nlp/2022-03-29-transformers-note-8.html)

运行 *run_summarization_mt5.sh* 脚本即可进行训练。

```
bash run_summarization_mt5.sh
```

如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 mT5 模型在测试集上的 ROUGE-1、ROUGE-2 和 ROUGE-L 值分别为 70.00、55.56 和 70.00 （Nvidia Tesla V100, batch=32）。
