import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig

class WeightedSumEmbeddingsClassifer(nn.Module):

    def __init__(self, vocab_size, hidden_size=768, padding_idx=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.embeddings_weight = torch.normal(0, 1, size=(self.vocab_size, self.hidden_size))
    
    def forward(self, src_ids):
        '''
        Input shape: (batch_size, sequence_length, vocab_size) after softmax!!!
        Output shape: (batch_size, sequence_length, hidden_size)
        '''
        return torch.matmul(src_ids, self.embeddings_weight)
        

class ClassifierBERT(nn.Module):

    def __init__(self, vocab_size, dropout):
        super().__init__()

        self.vocab_size = vocab_size
        self.config = DistilBertConfig(
            vocab_size=vocab_size,
            seq_classif_dropout=dropout
        )

        self.BERT = DistilBertForSequenceClassification(self.config).from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        self.embeddings = WeightedSumEmbeddingsClassifer(self.vocab_size)
    
    def forward(self, src_ids):
        embeddings = self.embeddings(src_ids)
        output = self.BERT(
            inputs_embeds=embeddings,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        # return is before softmax
        return output.logits
