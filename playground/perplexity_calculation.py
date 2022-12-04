from transformers import GPT2LMHeadModel

import torch
import torch.nn as nn
from scipy.special import softmax

class WeightedSumEmbeddingsPerplexity(nn.Module):
    def __init__(self, vocab_size, hidden_size=1280, padding_idx=0):
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.padding_idx=padding_idx
        self.embeddings_weight=torch.normal(0, 1, size=(self.vocab_size, hidden_size))
    
    def forward(self, logits):
        # logits: (batch_size, 128, vocab_size)
        # return: (batch_size, 128, hidden_size)
        return torch.matmul(logits, self.embeddings_weight) #after softmax

def perplexity(logits, vocab_size):
    device = "cuda"
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    logits=softmax(logits, axis=2)
    logits=torch.from_numpy(logits)
    logits_to_embedding=WeightedSumEmbeddingsPerplexity(vocab_size=vocab_size)

    trg_len=128
    with torch.no_grad():
        weighted_embeds=logits_to_embedding.forward(logits)
        outputs = model(inputs_embeds=weighted_embeds, labels=weighted_embeds)
        # nll = outputs.loss
        output_logits=outputs.logits
        #### TODO: Calculate Loss

    ppl = torch.exp(nll/128)
    return ppl
    
# perplexity(logits)