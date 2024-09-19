### Seq2Seq任务（翻译）

详细说明请见[《Transformers 库快速入门 第十章：翻译任务》](https://transformers.run/c3/2022-03-24-transformers-note-7/)

运行 *run_translation_marian.sh* 脚本即可进行训练。

```
bash run_translation_marian.sh
```

如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_pred`。

> 经过 3 轮训练，最终 Marian 模型在测试集上的 BLEU 值为 52.96（Nvidia Tesla V100, batch=32）。