### 序列标注任务（抽取式问答）

详细说明请见[《Hugging Face 的 Transformers 库快速入门（九）：抽取式问答》](https://xiaosheng.run/2022/04/02/transformers-note-9.html)

运行 *run_extractiveQA.sh* 脚本即可进行训练。

```
bash run_extractiveQA.sh
```

如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的 F1 和 EM 值分别为 67.96 和 31.84 （Nvidia Tesla V100, batch=4）。