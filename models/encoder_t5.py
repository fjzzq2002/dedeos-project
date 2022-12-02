import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Config

class EncoderT5(nn.Module):
    '''
    An encoder that uses the Hugging Face T5-small V1.1 encoder pretrained parameters
    '''

    def __init__(self, vocab_size):
        '''
        See doc for T5Config and T5EncoderModel at:
        https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config
        https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel
        '''
        super().__init__()

        self.vocab_size = vocab_size
        self.config = T5Config(
            vocab_size=vocab_size,
            feed_forward_proj='gated-gelu'
        )

        self.T5 = T5EncoderModel(self.config).from_pretrained('google/t5-v1_1-small')
        self.ff = nn.Sequential( # Using sequential to leave space for expansion
            nn.Linear(512, self.vocab_size)
        )
    
    def forward(self, src_ids):
        '''
        Input size: (batch_size, sequence_length), tokenized sequences in the batch
        Output size: (batch_size, sequence_length, vocab_size), un-normalized logits 
        for each position for each sequence
        '''
        outputs = self.T5(
            input_ids=src_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        logits = self.ff(outputs.last_hidden_state)
        return logits