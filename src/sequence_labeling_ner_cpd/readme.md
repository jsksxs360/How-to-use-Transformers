### 序列标注任务（命名实体识别）

详细说明请见[《Hugging Face 的 Transformers 库快速入门（六）：序列标注任务》](https://xiaosheng.run/2022/03/18/transformers-note-6.html)

与 Transformers 库类似，我们将模型损失的计算也包含进模型本身，这样在训练循环中我们就可以直接使用模型返回的损失进行反向传播。

为了简化数据处理，这里我们并没有将 `[CLS]`、`[SEP]`、`[PAD]` 等特殊 token 对应的标签设为 -100，而是维持原始的 0 值，然后在计算损失时借助 Attention Mask 来排除填充位置：

```python
active_loss = attention_mask.view(-1) == 1
active_logits = logits.view(-1, self.num_labels)[active_loss]
active_labels = labels.view(-1)[active_loss]
loss = loss_fct(active_logits, active_labels)
```

最后通过 `view()` 将 batch 中的多个向量序列连接为一个序列，这样就可以直接使用交叉熵函数计算损失，而不必进行维度调整。

除了本文介绍的纯基于 BERT 的 NER 模型，我们还实现了一个带有 CRF 层的 BERT+CRF 模型，分别通过运行 *run_ner_softmax.sh* 和 *run_ner_crf.sh* 脚本进行训练。如果要进行测试或者将预测结果保存到文件，只需把脚本中的 `--do_train` 改成 `--do_test` 或 `--do_predict`。

> 经过 3 轮训练，最终 BERT 模型在测试集上的准确率为 95.10%，BERT+CRF 为 95.37%（Nvidia Tesla V100, batch=4）。