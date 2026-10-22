import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,d,heads,dropout):
        super().__init__(); self.heads=heads; self.d=d
        self.ln1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.ln2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(approximate="tanh"),nn.Linear(4*d,d),nn.Dropout(dropout)); self.drop=nn.Dropout(dropout)
    def forward(self,x,valid):
        b,t,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).view(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        causal=torch.ones(t,t,dtype=torch.bool,device=x.device).tril()
        allowed=causal[None,None,:,:] & valid[:,None,None,:]
        # Padding queries may attend to themselves, avoiding all-masked softmax rows.
        allowed=allowed | ((~valid)[:,None,:,None] & torch.eye(t,dtype=torch.bool,device=x.device)[None,None])
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=self.drop.p if self.training else 0.)
        x=x+self.drop(self.proj(y.transpose(1,2).reshape(b,t,d)))
        return x+self.ff(self.ln2(x))

class GPT2Rec(nn.Module):
    def __init__(self,catalog_size,history_length,hidden=128,layers=2,heads=4,dropout=.1):
        super().__init__(); self.catalog_size=catalog_size; self.max_length=history_length+20
        self.item=nn.Embedding(catalog_size+2,hidden,padding_idx=0)
        self.position=nn.Embedding(self.max_length,hidden)
        self.blocks=nn.ModuleList([Block(hidden,heads,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(hidden); self.drop=nn.Dropout(dropout)
        self.apply(self.initialize)
        with torch.no_grad(): self.item.weight[0].zero_()
    @staticmethod
    def initialize(m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=.02)
        if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self,history):
        valid=history.ne(0); positions=(valid.long().cumsum(-1)-1).clamp(min=0,max=self.max_length-1)
        x=self.drop(self.item(history)+self.position(positions))
        for block in self.blocks: x=block(x,valid)
        return self.norm(x[:,-1])
    def forward(self,history):
        return F.linear(self.encode(history),self.item.weight[1:self.catalog_size+1])
