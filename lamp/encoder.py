from ..data.rcv1 import RCV1DS
import torch.nn as nn


class Enc(nn.Module):
    def __init__(self,v=1e4,embed_d=256,h=256,dropout=0):
        super(Enc,self).__init__()
        self.emb=nn.Embedding(v+2,embed_d,padding_idx=0)
        self.gru=nn.GRU(embed_d,h,dropout=dropout)
    def forward(self,x):
        x=self.emb(x)
        x,_=self.gru(x)
        return x.mean(axis=0)
        
        