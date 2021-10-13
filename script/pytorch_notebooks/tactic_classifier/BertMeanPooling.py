from transformers import BertTokenizer, BertConfig, BertModel, AdamW
import utils
import torch
import torch.nn as nn
from SimpleClassifier import SimpleClassifier
from MLPClassifier import MLPClassifier


class BertMeanPooling(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertMeanPooling(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, BERT_MODEL='bert-base-uncased'):
        super(BertMeanPooling, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL, gradient_checkpointing=True)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def initialize_classifier(self, device, use_hidden_layer=False, hidden_size=0,
                              num_labels=2):
        if use_hidden_layer:
            self.classifier = MLPClassifier(self.bert.config.hidden_size, hidden_size, num_labels)
        else:
            self.classifier = SimpleClassifier(self.bert.config.hidden_size, num_labels)

        self.classifier = utils.to(self.classifier, device)

    def forward(self, document_batch: torch.Tensor, batch_size, device):
        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0], 1,
                                        self.bert.config.hidden_size),
                                  dtype=torch.float)

        # only pass through bert_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_batch.shape[0]):
            input_ids = utils.to(document_batch[doc_id][:, 0], device)
            token_type_ids = utils.to(document_batch[doc_id][:, 1], device)
            attention_mask = utils.to(document_batch[doc_id][:, 2], device)
            bert_op = self.bert(input_ids,
                                token_type_ids,
                                attention_mask)
            pooled_output = self.dropout(bert_op[1])
            bert_output[doc_id] = pooled_output.mean(dim=0)

        max_ = utils.to(bert_output.mean(dim=1), device)
        prediction = self.classifier(max_)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def print_model_parameters(self):
        # Get all of the model's parameters as a list of tuples.
        params = list(self.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(
            len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))