import re
import torch
from torch.utils import data
from collections import Counter
from tqdm import tqdm

# Took me 2 min to run

def stop(limit = 1, times = [0]):
    times[0] += 1
    assert times[0] < limit, "STOP HERE"


class Tokenizer(object):

    def __init__(self, docs, PARAGRAPH_LENGTH=128, UNK_THRESHOLD=20) -> None:
        
        super().__init__()

        SEP_TOKEN = "<SEP>"
        SOS_TOKEN = "<SOS>" # This is a misnomer, it is actually the first token of the paragraph
        EOS_TOKEN = "<EOS>" # This is a misnomer, it is actually the last token of the paragraph

        def combine_and_cut(rawtext):
            """
            Generates finer grained sentences by splitting on punctuation.
            Used to cut the document into paragraphs.

            Input: list of strings
            Output: Generate sentences one by one
            """
            for sent in rawtext:
                # Split on punctuation
                sent = re.split(r' ([.!?])', sent)
                # Remove punctuation at the beginning and end of sentences
                sent = [re.sub(r'^[.!?]', '', s) for s in sent]
                sent = [re.sub(r'[.!?]$', '', s) for s in sent]
                # Remove empty strings and convert to lower case
                sent = [x.lower() for x in sent if x]
                yield from sent
        
        #---------------------------------------------------------------------------   
        #  Cut the documents into paragraphs of PARAGRAPH_LENGTH
        #---------------------------------------------------------------------------     
        doc_texts = [] # List of paragraphs
        gender_label = []

        print(f"Cutting documents into paragraphs of length {PARAGRAPH_LENGTH}...")
        for doc in tqdm(docs):

            gender = int(doc['gender'] == 'male')
            doc_texts.append(SOS_TOKEN)
            gender_label.append(gender)

            for sent in combine_and_cut(doc['rawtext']):
                if len(doc_texts[-1].split()) + len(sent.split()) > PARAGRAPH_LENGTH - 2: # Minus 2 for SEP and EOS tokens
                    doc_texts[-1] += " " + EOS_TOKEN
                    doc_texts.append(SOS_TOKEN)
                    gender_label.append(gender)
                doc_texts[-1] += sent + " " + SEP_TOKEN
            doc_texts[-1] += " " + EOS_TOKEN
        
        print(f"Number of documents: {len(doc_texts)}")

        print(f"Counting freqeuncies of words...")
        freq = Counter() # Count the number of times each word appears
        doc_texts_token = [] # This is a list of lists of tokens
        for doc_text in tqdm(doc_texts):
            if len(doc_text.split()) <= PARAGRAPH_LENGTH:
                doc_text_token = doc_text.split()
                freq.update(doc_text_token)
                doc_texts_token.append(doc_text_token)
        
        print(f"Number of documents with lengths <= {PARAGRAPH_LENGTH}: {len(doc_texts_token)}")

        #---------------------------------------------------------------------------   
        #  Convert words to <UNK>, then to indices
        #---------------------------------------------------------------------------     
        print("Number of unique words before converting to <UNK>: ", len(freq))
        before_occur = sum(freq.values())

        unique_words = set()

        print(f"Converting words with frequencies less than {UNK_THRESHOLD} to <UNK>...")
        total_occur = before_occur
        for i, doc_text_token in tqdm(enumerate(doc_texts_token)):
            # Replace words with less than 5 occurrences with <UNK>
            doc_text_token = [word if freq[word] > UNK_THRESHOLD else "<UNK>" for word in doc_text_token]
            unique_words.update(doc_text_token)
            total_occur -= doc_text_token.count("<UNK>")
            doc_texts_token[i] = doc_text_token
        print("Number of unique words after converting <UNK>: ", len(unique_words))
        print(f"Known occurrences rate {round(total_occur/before_occur * 100, 2)}%")

        self._vocab_size = len(unique_words) # numbers of unique words == len(token2idx)
        self._vocab = unique_words # set of unique words
        self._token2idx = {token: idx for idx, token in enumerate(unique_words)} # dict of unique words to indices
        self._idx2token = {idx: token for idx, token in enumerate(unique_words)} # dict of indices to unique words
    
    def token2idx(self, token):
        return self._token2idx[token] if token in self._token2idx else self._token2idx["<UNK>"]
    
    def idx2token(self, idx):
        return self._idx2token[idx]

    def vocab_size(self):
        return self._vocab_size


class GenderDataset(data.Dataset):

    def __init__(self, docs, tokenizer, PARAGRAPH_LENGTH = 128) -> None:
        super().__init__()
        
        SEP_TOKEN = "<SEP>"
        SOS_TOKEN = "<SOS>" # This is a misnomer, it is actually the first token of the paragraph
        EOS_TOKEN = "<EOS>" # This is a misnomer, it is actually the last token of the paragraph

        def combine_and_cut(rawtext):
            """
            Generates finer grained sentences by splitting on punctuation.
            Used to cut the document into paragraphs.

            Input: list of strings
            Output: Generate sentences one by one
            """
            for sent in rawtext:
                # Split on punctuation
                sent = re.split(r' ([.!?])', sent)
                # Remove punctuation at the beginning and end of sentences
                sent = [re.sub(r'^[.!?]', '', s) for s in sent]
                sent = [re.sub(r'[.!?]$', '', s) for s in sent]
                # Remove empty strings and convert to lower case
                sent = [x.lower() for x in sent if x]
                yield from sent
        
        #---------------------------------------------------------------------------   
        #  Cut the documents into paragraphs of PARAGRAPH_LENGTH
        #---------------------------------------------------------------------------     
        doc_texts = [] # List of paragraphs
        gender_label = []

        print(f"Cutting documents into paragraphs of length {PARAGRAPH_LENGTH}...")
        for doc in tqdm(docs):

            gender = int(doc['gender'] == 'male')
            doc_texts.append(SOS_TOKEN)
            gender_label.append(gender)

            for sent in combine_and_cut(doc['rawtext']):
                if len(doc_texts[-1].split()) + len(sent.split()) > PARAGRAPH_LENGTH - 2: # Minus 2 for SEP and EOS tokens
                    doc_texts[-1] += " " + EOS_TOKEN
                    doc_texts.append(SOS_TOKEN)
                    gender_label.append(gender)
                doc_texts[-1] += sent + " " + SEP_TOKEN
            doc_texts[-1] += " " + EOS_TOKEN
        
        print(f"Number of documents: {len(doc_texts)}")

        print(f"Counting freqeuncies of words...")
        freq = Counter() # Count the number of times each word appears
        doc_texts_token = [] # This is a list of lists of tokens
        new_gender_label = []
        for doc_text, label in tqdm(zip(doc_texts, gender_label)):
            if len(doc_text.split()) <= PARAGRAPH_LENGTH:
                doc_text_token = doc_text.split()
                freq.update(doc_text_token)
                doc_texts_token.append(doc_text_token)
                new_gender_label.append(label)
        
        print(f"Number of documents with lengths <= {PARAGRAPH_LENGTH}: {len(doc_texts_token)}")
        
        self.raw_text = doc_texts # list of strings of close to PARAGRAPH_LENGTH words
        self.data = [[tokenizer.token2idx(token) for token in doc] for doc in doc_texts_token]
        self.label = new_gender_label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    

    def tokenize(self, idx):
        """
        Converts indice or a list of indices to tokens.
        Input: int or list of ints
        Output: string or list of strings
        """
        if isinstance(idx, int):
            return self.idx2token[idx]
        elif isinstance(idx, list):
            return " ".join([self.idx2token[i] for i in idx])
        else:
            raise TypeError(f"Expected int or list, got {type(idx)}")
    
    def detokenize(self, token):
        """
        Converts token or a list of tokens to indices.
        Input: string. Document/paragraph to be converted to indices
        Output: list of ints
        """
        if isinstance(token, str):
            return [self.token2idx[t] for t in token.split()]
        else:
            raise TypeError(f"Expected str, got {type(token)}")

def gender_data_collate_fn(gender_data):
    src_len = torch.tensor([len(gender_datum[0]) for gender_datum in gender_data], dtype=torch.int32)
    max_len = max(src_len)
    src_ids = torch.stack([torch.cat([
        torch.tensor(gender_datum[0], dtype=torch.int32), 
        torch.zeros(max_len - len(gender_datum[0]), dtype=torch.int32)
        ]) for gender_datum in gender_data])
    tgt = torch.tensor([gender_datum[1] for gender_datum in gender_data])
    return src_ids, src_len, tgt