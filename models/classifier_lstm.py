import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ClassifierLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, is_bidirectional):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size
        )
        self.rnn = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout=0.2,
            bidirectional=self.is_bidirectional,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size * self.num_layers * 2 * (2 if self.is_bidirectional else 1), 2)

    def forward(self, src_ids, src_len):
        embed = self.embedding(src_ids)
        packed_src = pack_padded_sequence(embed, src_len, batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.rnn(packed_src)
        output = torch.cat([
            hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1), 
            cell.permute(1, 0, 2).contiguous().view(cell.size(1), -1)
            ], dim=1)
        logits = self.linear(output)
        return logits