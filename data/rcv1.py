import chazutsu
import json
from os import path
from tqdm import tqdm

def create_vocab_with_count(chazutsu_ressource,filename):
    count={}
    for t in tqdm(chazutsu_ressource.test_data()['words']):
        t=t.split(' ')
        for w in t:
            if w not in count:
                count[w]=1
            else:
                count[w]+=1
    for t in tqdm(chazutsu_ressource.data()['words']):
        t=t.split(' ')
        for w in t:
            if w not in count:
                count[w]=1
            else:
                count[w]+=1

    words=list(count.keys())
    words.sort(key=lambda w: count[w],reverse=True)
    
    with open(filename,'w+') as fd:
        json.dump(words,fd)
class RCV1DS:
    def __init__(self,vocab_size,filename=None):
        if not filename:
            filename=
        
if __name__=='__main__':           
    data=chazutsu.datasets.ReutersNews().download()
    create_vocab_with_count(data,'vocab.json')
