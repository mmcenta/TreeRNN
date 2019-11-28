import os.path as path
from time import time
import torch

def read_rcv1_files():
    ##returns dict containing associating ids and x: list of training tokens and y: list of tags
    file_dir=path.dirname(path.realpath(__file__))
    file_list=[path.join(file_dir,f'lyrl2004_tokens_test_pt{i}.dat') for i in range(4)]
    file_list.append(path.join(file_dir,f'lyrl2004_tokens_train.dat') )
    
    data={}
    vocab={}
    vocab_idx=0
    top_vocab={}
    
    with open(path.join(file_dir,'rcv1_topics.dat'),'r') as fd:
        for i,top in enumerate(fd):
            if top!='\n':
                top_vocab[top[:-1]]=i
            
    for filename in file_list:
        with open(filename,'r') as fd:
            idx=None
            aux_lines=[]
            for line in fd:
                if not idx and line.startswith('.I'):
                    idx=line[3:-1]
                    aux_lines=[]
                elif line!='\n' and not line.startswith('.W'):
                    aux_lines.append(line[:-1])
                elif line=='\n':
                    aux=[tok for l in aux_lines for tok in l.split()]
                    data[idx]=[[],[]]
                    for tok in aux:
                        if tok in vocab:
                            vocab[tok][1]+=1
                        else:
                            vocab[tok]=[vocab_idx,1]
                            vocab_idx+=1
                        data[idx][0].append(vocab[tok][0])
                                                
                    idx=None
            
    
    with open(path.join(file_dir,'rcv1-v2.topics.qrels'),'r') as fd:
        for l in fd:
            l=l.split()
            data[l[1]][1].append(top_vocab[l[0]])
          
        
    return list(data.values()),vocab,top_vocab,list(data.keys())

def revoc_data(data,vocab,n_tok):
    aux=list(vocab.keys())
    aux.sort(key = lambda w :vocab[w][1],reverse=True)
    aux=aux[:n_tok]
    ivocab={v[0]:k for k,v in vocab.items()}
    for i in range(100):
        print(str(i)+' '+ivocab.get(i,'No voc'))
    res_voc={k:v+2 for v,k in enumerate(aux)}
    for t in data:
        for i in range(len(t[0])):
            aux=res_voc.get(ivocab[int(t[0][i])],1)
            t[0][i]=aux
    return data,res_voc

class RCV1DS:
    ##0 reserved to pad_token
    ##1 corresponds to unknown token
    def __init__(self,n_tok=1000):
        data,vocab,self.top_vocab,self.ids=read_rcv1_files()
        self.data,self.vocab=revoc_data(data,vocab,n_tok)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return torch.tensor(self.data[idx][0]),torch.tensor(self.data[idx][1])
        
        

if __name__=='__main__':
    start=time()
    ds=RCV1DS()
    print(time()-start)
    print(ds[13])