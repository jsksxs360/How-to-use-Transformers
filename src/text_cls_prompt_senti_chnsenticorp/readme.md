### 文本分类任务（Prompting 情感分析）

详细说明请见[《Transformers 库快速入门 第十三章：Prompting 情感分析》](https://transformers.run/nlp/2022-10-10-transformers-note-10.html)

运行 *run_prompt_senti_bert.sh* 脚本即可进行训练。

```bash
bash run_prompt_senti_bert.sh
```

如果要进行测试或者将模型输出的翻译结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的结果为：
>
> ```
> ==> Nvidia GeForce RTX 3090, batch=4, vtype=base
> POS: 96.48 / 94.74 / 95.60, NEG: 94.69 / 96.45 / 95.56
> micro_F1 - 95.5835 macro_f1 - 95.5833
> 
> ==> Nvidia GeForce RTX 3090, batch=4, vtype=virtual
> POS: 96.79 / 94.08 / 95.41, NEG: 94.09 / 96.79 / 95.42
> micro_F1 - 95.4166 macro_f1 - 95.4167
> ```
