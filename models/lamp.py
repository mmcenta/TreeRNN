import torch
import torch.nn as nn
from message_passing import MessagePassing
from inspect import getfullargspec
getarg=getfullargspec

    
class SelfAttention(nn.Module):
    def __init__(self,d):
        super(SelfAttention,self).__init__()
        self.Q=nn.Linear(d,d)
        self.K=nn.Linear(d,d)
    def forward(self,x):
        """
            OUTPUT: (batch_size,n,n)
        """
        queries=self.Q(x)
        keys=self.K(x)
        batch_size, output_len, d = queries.size()

        att_scores=torch.bmm(queries,keys.transpose(1,2).contiguous())
        att_scores = att_scores.view(batch_size,output_len,output_len)
        return torch.exp(att_scores)



class LAMP(MessagePassing):
    def __init__(self,embed_dim,message_dim):
        super(LAMP, self).__init__(aggr='add')  # "Add" aggregation.
        self.att_layer=SelfAttention(embed_dim)
        self.message_linear=nn.Linear(embed_dim,message_dim)
        self.update_mlp=nn.Linear(message_dim,embed_dim)
    
    def forward(self,x,edge_index):
        """
        Args:
            X: torch.tensor (batch_size,num_nodes,dim)
        """
        num_nodes,d,device=x.shape[0],x.shape[1],x.device
        adj=[[] for _ in range(num_nodes)]
        for idx in range(edge_index.size(1)):
            adj[edge_index[0,idx]].append(edge_index[1,idx])
        
        
        adj=[torch.tensor(t,device=device) for t in adj]
        
        
        att_scores=self.att_layer(x) ##(batch_size,n,n)
        att_sum=torch.cat([self.att_scores[:,i,adj[i]].sum(axis=2).view(-1).unsqueeze(1) \
                                   for i in range(num_nodes)],axis=1) ##(batch_size,n)
        att_scores=att_scores/att_sum
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=self.message_linear(x),
                              att_scores=att_scores,x_old=x)
    
    def message(self,x_j,edge_index,att_scores):
        return att_scores[edge_index[0],edge_index[1]]*x_j
    
    def update(self,aggr_out,x_old):
        return x_old+self.update_mlp(aggr_out)
        


class TestMP(MessagePassing):
    def __init__(self):
        super(TestMP,self).__init__()
        print(getarg(self.message)[0])
    def message(self,x_j,edge_index):
        print(x_j.shape)
        print(edge_index.shape)
        return None
    def forward(self,x,edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

def test():
    x=torch.randn((10,50)).unsqueeze(0)
    edge_index=torch.tensor([list(range(10)),list(range(1,10))+[0]])
    mp=LAMP(50,50)
    mp(x,edge_index)
if __name__=='__main__':
    test()        