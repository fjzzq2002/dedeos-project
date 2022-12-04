import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer('all-mpnet-base-v2')

def encode_for_similarities(st_model, device, src_ids, rst_logits):

    SEP=st_model.tokenizer.sep_token_id
    CLS=st_model.tokenizer.cls_token_id
    # print(CLS,SEP)
    cls_emb=st_model[0].auto_model.embeddings.word_embeddings.weight[torch.ones(src_ids.shape[0], 1, dtype=torch.long).to(device) * CLS].to(device)
    sep_emb=st_model[0].auto_model.embeddings.word_embeddings.weight[torch.ones(src_ids.shape[0], 1, dtype=torch.long).to(device) * SEP].to(device)
    # print(cls_emb)
    
    src_full = torch.cat([torch.ones(src_ids.shape[0], 1, dtype=torch.long).to(device) * CLS,
                src_ids,
                torch.ones(src_ids.shape[0], 1, dtype=torch.long).to(device) * SEP], dim=1)
    
    src_embed = st_model[0].auto_model.embeddings.word_embeddings.weight[src_full].to(device)
    tgt_embed = nn.Softmax(dim=-1)(rst_logits)@st_model[0].auto_model.embeddings.word_embeddings.weight.to(device)
    tgt_embed = torch.cat([cls_emb, tgt_embed, sep_emb], dim=1)
    
    src_encode = st_model[0].auto_model.forward(inputs_embeds=src_embed)[1]
    tgt_encode = st_model[0].auto_model.forward(inputs_embeds=tgt_embed)[1]
    
    # normalize with torch.nn.functional.normalize
    src_encode = torch.nn.functional.normalize(src_encode, p=2, dim=1)
    tgt_encode = torch.nn.functional.normalize(tgt_encode, p=2, dim=1)
    #print(src_encode.shape, tgt_encode.shape)
    #print(src_encode-tgt_encode)
    # print(result.shape)
    # print(result)

    return src_encode, tgt_encode