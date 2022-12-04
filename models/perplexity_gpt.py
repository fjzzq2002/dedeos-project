import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class WeightedSumEmbeddingsPerplexity(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, padding_idx=0):

        super().__init__()

        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.padding_idx=padding_idx
        self.embeddings = nn.Embedding(self.vocab_size, hidden_size, self.padding_idx)
        # self.embeddings_weight=torch.normal(0, 1, size=(self.vocab_size, hidden_size))
    
    def forward(self, logits):
        # logits: (batch_size, 128, vocab_size)
        # return: (batch_size, 128, hidden_size)
        return torch.matmul(logits, self.embeddings.weight) #after softmax

class PerplexityGPT2(nn.Module):

    def __init__(self, vocab_size, bos_token_id, eos_token_id):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.config = GPT2Config.from_pretrained(
            "gpt2",
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id
        )

        self.GPT2 = GPT2LMHeadModel(self.config)
        self.embeddings = WeightedSumEmbeddingsPerplexity(self.vocab_size)
    
    def forward(self, src_logits):
        '''
        Returns the logits of the language model output, needs to cross entropy with
        the original labels or the logits of labels to get perplexity
        '''
        embeddings = self.embeddings(src_logits)
        output = self.GPT2(
            inputs_embeds=embeddings,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        # return is before softmax
        return output.logits