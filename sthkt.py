import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
from typing import Union,Callable,Optional,Any
from torch import Tensor
from enum import IntEnum
from spatial import MyGraph
from rw import MyRWGraph
import config as cg
import numpy as np

device=cg.DEVICE

class HGAKT(nn.Module):
    def __init__(self,d_model,dropout,final_fc_dim=512,n_heads=cg.HEADNUM,d_ffn=2048,l2=1e-5,preG=False,hawkes=False):
        super(HGAKT,self).__init__()

        self.d_model=d_model
        self.n_q=cg.NUM_Q
        self.n_kc=cg.NUM_KC
        self.n_stu=cg.NUM_STU
        self.dropout=dropout
        self.n_heads=n_heads
        self.d_ffn=d_ffn
        self.l2=l2

        self.hawkes=hawkes

        if preG is True:
            # m1=torch.load(cg.PREG_EMB_Q_N).to(device)
            # m2=torch.load(cg.PREG_EMB_Q_P).to(device)
            # q_emb_matrix=torch.concat((m1[:-1,:],m2),dim=0)
            # self.q_emb_layer=nn.Embedding(self.n_q+1,self.d_model,_weight=(m1+m2)/2)
            # self.qa_emb_layer=nn.Embedding(self.n_q*2+1,self.d_model,_weight=q_emb_matrix)
            self.q_emb_layer=nn.Embedding(self.n_q+1,self.d_model,_weight=torch.load(cg.PREG_EMB_Q).to(device))
            self.a_emb_layer=nn.Embedding(3,self.d_model,_weight=torch.load(cg.PREG_EMB_A).to(device))
            self.kc_emb_layer=nn.Embedding(self.n_kc+1,self.d_model,_weight=torch.load(cg.PREG_EMB_KC).to(device))
            self.s_emb_layer=nn.Embedding(self.n_stu,self.d_model,_weight=torch.load(cg.PREG_EMB_S).to(device))
        else:
            self.q_emb_layer=nn.Embedding(self.n_q+1,self.d_model)
            self.a_emb_layer=nn.Embedding(3,self.d_model)
            # self.qa_emb_layer=nn.Embedding(self.n_q*2+1,self.d_model)
            self.kc_emb_layer=nn.Embedding(self.n_kc+1,self.d_model)
            self.s_emb_layer=nn.Embedding(self.n_stu+1,self.d_model)
            print(self.s_emb_layer.weight.data.shape,"Look")

        self.q_Transformer_layer=TransformerLayer(d_model=d_model,d_feature=d_model//n_heads,d_ffn=d_ffn,dropout=dropout,n_heads=n_heads)
        self.qa_Transformer_layer=TransformerLayer(d_model=d_model,d_feature=d_model//n_heads,d_ffn=d_ffn,dropout=dropout,n_heads=n_heads)
        self.qqa_Transformer_layer=TransformerLayer(d_model=d_model,d_feature=d_model//n_heads,d_ffn=d_ffn,dropout=dropout,n_heads=n_heads,hawkes=hawkes)

        if hawkes is True:
            if cg.ABLATION=='withoutConv':
                print(cg.ABLATION)
                self.graph=MyRWGraph(d_model,self.q_emb_layer.weight.data,self.kc_emb_layer.weight.data,\
                                self.s_emb_layer.weight.data,self.a_emb_layer.weight.data)
            else:
                self.graph=MyGraph(d_model,self.q_emb_layer.weight.data,self.kc_emb_layer.weight.data,\
                                self.s_emb_layer.weight.data,self.a_emb_layer.weight.data)

        self.layernorm_layer_q=nn.LayerNorm(d_model)
        self.layernorm_layer_qa=nn.LayerNorm(d_model)
        self.layernorm_layer_h=nn.LayerNorm(d_model)

        self.out=nn.Sequential(
            nn.Linear(d_model*2,final_fc_dim),
            nn.ReLU(),
            nn.Linear(final_fc_dim,d_model),
            nn.Dropout(self.dropout),
        )

        self.result_layer=nn.Sequential(
            nn.Linear(d_model*2, 256), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256,1)
        )
        
        # self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_q+1 and self.n_q > 0:
                torch.nn.init.constant_(p, 0.)


    def forward(self,q_data,a_data):
        # torch.save(self.q_emb_layer.weight.data,"emb_src.pth")
        # q_data:[bs,seqlen]
        q_embed_data=self.q_emb_layer(q_data) #[bs,seqlen,dmodel]

        qa_data=q_data+a_data*self.n_q
        qa_data=torch.where(qa_data<self.n_q*2, qa_data, torch.tensor(self.n_q*2).to(device))
        # qa_embed_data=self.qa_emb_layer(qa_data)
        a_embed_data=self.a_emb_layer(a_data)
        qa_embed_data=q_embed_data+a_embed_data

        x=self.q_Transformer_layer(mask=1,query=q_embed_data,key=q_embed_data,values=q_embed_data)
        x=self.layernorm_layer_q(x)
        y=self.qa_Transformer_layer(mask=1 ,query=qa_embed_data,key=qa_embed_data,values=qa_embed_data) #[B,S,E]
        y=self.layernorm_layer_qa(y)

        #####################Get Hawkes Process Result#########################
        hq_qkc=None
        hq_qs=None
        if self.hawkes is True:
            hq_qkc,hq_qs=self.graph(q_embed_data,y,q_data)
        #######################################################################

        h=self.qqa_Transformer_layer(mask=0,query=x,key=x,values=y,h_qkc=hq_qkc,h_qs=hq_qs)
        # h=self.layernorm_layer_h(h)

        result=self.result_layer(torch.cat((h,q_embed_data),dim=-1))
        result=torch.sigmoid(result)
        return result


class TransformerLayer(nn.Module):
    def __init__(self,d_model,d_feature,d_ffn,n_heads,dropout,hawkes=False):
        super(TransformerLayer,self).__init__()

        self.masked_attn_head=MultiHeadAttention(d_model,d_feature,n_heads,dropout,hawkes=hawkes)

        self.norm1_layer=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)

        self.linear1=nn.Linear(d_model,d_ffn)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ffn,d_model)

        self.norm2_layer=nn.LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)


    def forward(self,mask,query,key,values,apply_pos=True,\
                h_qkc=None,h_qs=None):
        seqlen,batch_size=query.shape[1],query.shape[0]
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True,\
                h_qkc=h_qkc,h_qs=h_qs)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False,\
                h_qkc=h_qkc,h_qs=h_qs)

        query = query + self.dropout1((query2))
        query = self.norm1_layer(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.norm2_layer(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_feature,n_heads,dropout,bias=True,hawkes=False):
        # print("MultiHeadAttention")
        super(MultiHeadAttention,self).__init__()

        self.d_model=d_model
        self.d_feature=d_feature
        self.n_heads=n_heads
        self.bias=bias

        self.v_linear=nn.Linear(d_model,d_model,bias=self.bias)
        self.k_linear=nn.Linear(d_model,d_model,bias=self.bias)
        self.dropout=nn.Dropout(dropout)
        self.out_proj=nn.Linear(d_model,d_model,bias=self.bias)
        
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self.hawkes=hawkes
        if hawkes is True:
            seqlen=cg.MAX_SEQ
            self.alpha=nn.Linear(seqlen,seqlen)
            self.alpha2=nn.Linear(seqlen,seqlen)
            self.theta=nn.Linear(seqlen,seqlen)
            self.beta=nn.Linear(seqlen,seqlen)

        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)

        constant_(self.k_linear.bias, 0.)
        constant_(self.v_linear.bias, 0.)
        constant_(self.out_proj.bias, 0.)

        if self.hawkes is True:
            xavier_uniform_(self.alpha.weight)
            xavier_uniform_(self.alpha2.weight)
            xavier_uniform_(self.theta.weight)
            xavier_uniform_(self.beta.weight)

    def forward(self,q,k,v,mask,zero_pad,\
                h_qkc=None,h_qs=None):

        batch_size=q.shape[0]
        seqlen=q.shape[1]

        k=self.k_linear(k).view(batch_size,-1,self.n_heads,self.d_feature)
        q=self.k_linear(q).view(batch_size,-1,self.n_heads,self.d_feature)
        v=self.v_linear(v).view(batch_size,-1,self.n_heads,self.d_feature)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2) #[bs,seqlen,head,dmodel//head]
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n","q\n",q,"\n")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n","v\n",v,"\n")
      
        scores=attention(q,k,v,self.d_feature,mask,self.dropout,zero_pad,self.gammas,hawkes=self.hawkes)
        torch.save(scores,cg.DATASET_ROOTPATH+"attnscore.pt")
        
        # print(scores)
        # hawkes=True 返回注意力分数，即[bs,heads,seqlen,seqlen]
        # hawkes=False 返回注意力分数*v，即[bs,heads,seqlen,dmodel]
                        
        if self.hawkes is True:
            hawkes_qkc=mask_cosine_similarity(h_qkc,h_qkc.transpose(1,2),hashead=True,head=cg.HEADNUM) #[bs,head,seqlen,seqlen]
            hawkes_qs=mask_cosine_similarity(h_qs,h_qs.transpose(1,2),hashead=True,head=cg.HEADNUM) #[bs,head,seqlen,seqlen]
            # print("hawkes_qkc=",hawkes_qkc)
            # print("hawkes_qs=",hawkes_qs.shape)
            dist=get_dist(batch_size,seqlen,hashead=True,head=cg.HEADNUM) #[bs,head,seqlen,seqlen]
            # print(hawkes_qkc.shape)
            # print(hawkes_qs.shape)
            # print(dist.shape)
            nopeek_mask = torch.triu( torch.ones((seqlen, seqlen)), diagonal=0).unsqueeze(0).repeat(batch_size,1,1)
            mask=(nopeek_mask==0).unsqueeze(1).repeat(1,cg.HEADNUM,1,1).to(device)
            hawkes_attenuation=self.beta(torch.log(dist))
            
            # print("torch.exp(-torch.log(dist)-hawkes_attenuation)=\n",torch.exp(-torch.log(dist)-hawkes_attenuation))
            if cg.ABLATION=="withoutHawkes":
                # print(cg.ABLATION)
                scores=scores+(self.alpha(hawkes_qkc)+self.alpha2(hawkes_qs))*mask.float()
            else:
                # scores=scores+((self.alpha2(hawkes_qkc))*torch.exp(-torch.log(dist)-hawkes_attenuation))*mask.float() #[bs,head,seqlen,seqlen]
                if cg.ABLATION=="nn":
                    scores=scores+(-hawkes_attenuation)*mask.float()
                elif cg.ABLATION=="qkc":
                    scores=scores+((self.alpha2(hawkes_qkc))*torch.exp(-torch.log(dist)-hawkes_attenuation))*mask.float()
                elif cg.ABLATION=="qs":
                    scores=scores+((self.alpha(hawkes_qs))*torch.exp(-torch.log(dist)-hawkes_attenuation))*mask.float()
                else:
                    scores=scores+((self.alpha(hawkes_qs)+self.alpha2(hawkes_qkc))*torch.exp(-torch.log(dist)-hawkes_attenuation))*mask.float()
            # scores=scores+self.alpha(hawkes_qkc)
            # print("scores2=",scores[0,0,:,:])
            # torch.save(scores,cg.DATASET_ROOTPATH+"attnscore_hawkes_nn.pt")
            scores=torch.matmul(scores, v)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) #[bs,seqlen,dmodel]
        output=self.out_proj(concat)

        return output

def get_dist(batch_size,seqlen,hashead=False,head=None):
    x1=torch.arange(seqlen).expand(seqlen,-1).to(device)
    x2=x1.transpose(0,1).contiguous().to(device)
    dist=torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor)+1e-5 #1,1,seqlen,seqlen
    dist=dist.to(device)
    if hashead is True:
        return dist.repeat(batch_size,head,1,1) #bs,head,seqlen,seqlen
    return dist

def mask_cosine_similarity(x,y,hashead=False,head=None):
    # x:[bs,seqlen,dmodel]
    # y:[bs,dmodel,seqlen]
    bs,seqlen,dmodel=x.shape
    similarity=torch.bmm(x,y)
    nsimilarity=similarity.to(device)
    # norm=torch.norm(x,dim=2,keepdim=True)*torch.norm(y,dim=1,keepdim=True).to(device)
    # nsimilarity=similarity/norm

    nopeek_mask = torch.triu( torch.ones((seqlen, seqlen)), diagonal=0).unsqueeze(0).repeat(bs,1,1)
    mask=(nopeek_mask==0).to(device)
    nsimilarity=nsimilarity.masked_fill(mask==0,-1e32)
    nsimilarity=F.softmax(nsimilarity,dim=-1)
    nsimilarity=nsimilarity*mask.float().to(device)

    if hashead is False:
        # 这种情况没写
        return nsimilarity #[bs,seqlen,seqlen]
    else:
        #默认
        return nsimilarity.unsqueeze(1).repeat(1,head,1,1) #[bs,head,seqlen,seqlen]

def head_cosine_similarity(head,x,y,dim):
    #x:[bs,seqlen,7*dmodel]
    temp=F.cosine_similarity(x.unsqueeze(2),y.unsqueeze(1),dim) #[bs,seqlen,seqlen]
    temp=temp.unsqueeze(1)
    # print("!!!!!!!!!!!!",temp.shape)
    result=temp.repeat(1,head,1,1) #[bs,head,seqlen,seqlen]
    return result

def attention(q,k,v,d_feature,mask,dropout,zero_pad,gamma=None,hawkes=False):
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_feature)

    batch_size,head,seqlen=scores.shape[0],scores.shape[1],scores.shape[2]

    x1=torch.arange(seqlen).expand(seqlen,-1).to(device)
    x2=x1.transpose(0,1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores=scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    scores=scores*mask.float().to(device)
    if zero_pad:
        pad_zero = torch.zeros(scores.shape[0], head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, :-1, :]], dim=2)
        
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if hawkes is True:
        return scores
    return output

if __name__=="__main__":
    transformer=TransformerLayer(128,32,512,4,0.1)
