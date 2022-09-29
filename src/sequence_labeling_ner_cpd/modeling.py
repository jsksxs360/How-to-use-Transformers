from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from ..tools import FullyConnectedLayer, CRF

class BertForNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            self.mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ffnn_size if args.use_ffnn_layer else config.hidden_size, self.num_labels)
        self.post_init()

    def forward(self, batch_inputs, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            attention_mask = batch_inputs.get('attention_mask')
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class BertCrfForNER(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            self.mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ffnn_size if args.use_ffnn_layer else config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()
    
    def forward(self, batch_inputs, labels=None):
        bert_output = self.bert(**batch_inputs)
        sequence_output = bert_output.last_hidden_state
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=batch_inputs.get('attention_mask'))
        return loss, logits
