import dgl
import dgl.function as fn
import networkx as nx
import torch
import torch.nn as nn
import config as cg
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.autograd import Variable
from preG import preG
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx

device=cg.DEVICE

class MyRWGraph(nn.Module):
    def __init__(self,d_model,E_q,E_kc,E_stu,preG=False):
        # print("MyGraph")
        super(MyRWGraph,self).__init__()

        self.d_model=d_model

        self.M_qkc=torch.load(cg.MATRIX_QKC).coalesce().to(device)
        self.M_kcs=torch.load(cg.MATRIX_KCS).coalesce().to(device)
        self.M_qs=torch.load(cg.MATRIX_QS).coalesce().to(device)
        
        self.emb_q=E_q.to(device)
        self.emb_kc=E_kc.to(device)
        self.emb_stu=E_stu.to(device)
        # self.emb_a=E_a.to(device)

        self.graph_data={
            ('question','contain','concept'):(torch.cat((self.M_qkc.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0),torch.cat((self.M_qkc.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0)),
            ('concept','dependOn','concept'):(torch.cat((self.M_kcs.indices()[0],torch.tensor([cg.NUM_KC]).to(device)),dim=0),torch.cat((self.M_kcs.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0)),
            ('student','finish','question'):(torch.cat((self.M_qs.indices()[1],torch.tensor([cg.NUM_STU]).to(device)),dim=0),torch.cat((self.M_qs.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0)),
            ('concept','contain-ed','question'):(torch.cat((self.M_qkc.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0),torch.cat((self.M_qkc.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0)),
            ('question','finish-ed','student'):(torch.cat((self.M_qs.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0),torch.cat((self.M_qs.indices()[1],torch.tensor([cg.NUM_STU]).to(device)),dim=0)),
        }

        self.g=dgl.heterograph(self.graph_data).to(device)
        self.set_feature(mode="graph",hasgrad=False)
        
        self.numN_q=self.g.num_nodes('question')
        self.numN_kc=self.g.num_nodes('concept')
        self.numN_stu=self.g.num_nodes('student')
        self.numE_qkc=self.g.num_edges('contain')
        self.numE_kcs=self.g.num_edges('dependOn')
        self.numE_sq=self.g.num_edges('finish')

        self.show_info()

    def set_feature(self,mode,hasgrad):
        if mode=='graph':
            self.g.nodes['question'].data["feature"]=self.emb_q
            # self.g.nodes['question'].data["feature-"]=self.emb_q
            self.g.nodes['concept'].data["feature"]=self.emb_kc
            self.g.nodes['student'].data["feature"]=self.emb_stu
        
        if mode=='qkc':
            self.g_qkc.nodes['question'].data["feature"]=self.emb_q
            # self.g_qkc.nodes['question'].data["feature-"]=self.emb_q
            self.g_qkc.nodes['concept'].data["feature"]=self.emb_kc
        
        if mode=='qs':
            self.g_qs.nodes['question'].data["feature"]=self.emb_q
            # self.g_qs.nodes['question'].data["feature-"]=self.emb_q
            self.g_qs.nodes['student'].data["feature"]=self.emb_stu

    def get_path(self,nodes,has_bs=True):
        #[bs,seqlen]
        if has_bs:
            bs,seq_len=nodes.shape
            nodes=torch.flatten(nodes)
        traces,types=dgl.sampling.random_walk(
            self.g,
            nodes,
            metapath=['contain', 'contain-ed','finish-ed', 'finish','contain', 'contain-ed','finish-ed', 'finish']
        )
        if has_bs:
            #[bs,seq_len,7]
            traces=torch.reshape(traces,[bs,seq_len,cg.CHAIN_LEN])
        return traces
    
    def get_path_emb(self,traces,mode=None):
        #[bs,seqlen,chainlen] -> [bs,seqlen,chainlen,d_model]
        Emb_Matrix_List=[]
        Emb_Matrix_List=['question','concept','question','student','question','concept','question','student','question']

        for i,emb_matrix in enumerate(Emb_Matrix_List):
            if i==0:
                emb=self.g.nodes[emb_matrix].data['feature'][traces[:,:,i]].unsqueeze(2)
            else:
                emb=torch.concat((emb,self.g.nodes[emb_matrix].data['feature'][traces[:,:,i]].unsqueeze(2)),dim=2)
        return emb
    
    def show_info(self):
        print(self.g.ntypes)
        print(self.g.etypes)
        print(self.g.canonical_etypes)
        print(self.g)
        print(self.g.metagraph().edges())
        print(self.g.num_nodes())
        print(self.g.num_nodes('question'))
        print(self.g.nodes('student'))   

    def forward(self,x_question,y_knowledge,seq_q):
        traces=self.get_path(seq_q) #[bs,seqlen,chainlen]
        traces_emb=self.get_path_emb(traces) #[bs,seqlen,chainlen,d_model]
        hq=traces_emb[:,:,0,:]+0.6*traces_emb[:,:,2,:]+0.4*traces_emb[:,:,4,:]+0.25*traces_emb[:,:,6,:]+0.1*traces_emb[:,:,8,:] #[bs,seqlen,dmodel]
        return hq,hq
