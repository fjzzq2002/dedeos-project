import torch
from torch.utils import data
from collections import Counter
from tqdm import tqdm
from sentence_transformers import CrossEncoder

tokenizer = CrossEncoder('cross-encoder/stsb-TinyBERT-L-4').tokenizer

# Took me 5 min to run

def stop(limit = 1, times = [0]):
    times[0] += 1
    assert times[0] < limit, "STOP HERE"


class GenderDataset(data.Dataset):

    def __init__(self, docs, PARAGRAPH_LENGTH=128, MIN_LENGTH=128, UNK_THRESHOLD=10) -> None:
        
        super().__init__()
        
        #---------------------------------------------------------------------------   
        #  Cut the documents into paragraphs of PARAGRAPH_LENGTH
        #---------------------------------------------------------------------------     
        doc_texts = [] # List of paragraphs
        gender_label = []

        print(f"Cutting documents into paragraphs of length {PARAGRAPH_LENGTH}...")
        freq = Counter() # Count the number of times each word appears
        discarded_doc = 0
        for doc in tqdm(docs):

            gender = int(doc['gender'] == 'male')
            # gender_label.append(gender)

            for text in (doc['rawtext']):
                tokens = self.str2token(text)
                i = 0
                while i + PARAGRAPH_LENGTH < len(tokens):
                    doc_texts.append(tokens[i : i + PARAGRAPH_LENGTH])
                    freq.update(doc_texts[-1])
                    gender_label.append(gender)
                    i += PARAGRAPH_LENGTH
                last_bit = tokens[-PARAGRAPH_LENGTH:]
                if len(last_bit) >= MIN_LENGTH:
                    doc_texts.append(last_bit)
                    gender_label.append(gender)
                    freq.update(doc_texts[-1])
                else:
                    discarded_doc += 1
        
        print(f"Number of documents: {len(doc_texts)}")
        print(f"Discarded ratio (due to MIN_LENGTH): {round((discarded_doc) / (discarded_doc + len(doc_texts)), 3)}")
        

        #---------------------------------------------------------------------------   
        #  Convert words to [UNK], then to indices
        #---------------------------------------------------------------------------     
        print("Number of unique words before converting to [UNK]: ", len(freq))
        before_occur = sum(freq.values())

        unique_words = set()

        ids = []

        print(f"Converting words with frequencies less than {UNK_THRESHOLD} to [UNK]...")
        total_occur = before_occur
        for i, doc_text in enumerate(tqdm(doc_texts)):
            # Replace words with less than 5 occurrences with [UNK]
            doc_text = [word if freq[word] > UNK_THRESHOLD else "[UNK]" for word in doc_text]
            unique_words.update(doc_text)
            total_occur -= doc_text.count("[UNK]")
            doc_texts[i] = doc_text
            ids.append(self.token2idx(doc_text))

        print("Number of unique words after converting [UNK]: ", len(unique_words))
        print(f"Known occurrences rate {round(total_occur/before_occur * 100, 2)}%")

        self._vocab_size = len(unique_words) # numbers of unique words == len(token2idx)
        self._vocab = unique_words # set of unique words
        self.raw_tokens = doc_texts # list of list of string tokens
        self.ids = ids # list of list of ints
        self.label = gender_label

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.label[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def str2idx(self, s: str):
        return tokenizer.encode(s)[1:-1]
    
    def str2token(self, s: str):
        return tokenizer.convert_ids_to_tokens(self.str2idx(s))
    
    def token2idx(self, tokens: list):
        return tokenizer.convert_tokens_to_ids(tokens)
    
    def idx2str(self, idx):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(idx))


def gender_data_collate_fn(gender_data):
    src_len = torch.tensor([len(gender_datum[0]) for gender_datum in gender_data], dtype=torch.int32)
    max_len = max(src_len)
    src_ids = torch.stack([torch.cat([
        torch.tensor(gender_datum[0], dtype=torch.int32), 
        torch.zeros(max_len - len(gender_datum[0]), dtype=torch.int32)
        ]) for gender_datum in gender_data])
    tgt = torch.tensor([gender_datum[1] for gender_datum in gender_data])
    return src_ids, src_len, tgt