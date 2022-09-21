### 句子对分类任务（金融语义相似度）

详细说明请见[《Hugging Face 的 Transformers 库快速入门（四）：微调预训练模型》](https://xiaosheng.run/2021/12/17/transformers-note-4.html)

与 Transformers 库类似，我们将模型损失的计算也包含进模型本身，这样在训练循环中我们就可以直接使用模型返回的损失进行反向传播。

这里我们同时加载 BERT 和 RoBERTa 权重来构建分类器，分别通过运行 *run_simi_bert.sh* 和 *run_simi_roberta.sh* 脚本进行训练。

```
bash run_simi_bert.sh
bash run_simi_roberta.sh
```

如果要进行测试或者将预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_pred`。

> 经过 3 轮训练，最终 BERT 在测试集（验证集）上的准确率为 73.61%，RoBERTa 为 73.84%（Nvidia Tesla V100, batch=16）。