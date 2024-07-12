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


class MyGraph(nn.Module):
    def __init__(self,d_model,E_q,E_kc,E_stu,preG=False):
        # print("MyGraph")
        super(MyGraph,self).__init__()

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
        # kc,question,stu都各有一个虚拟节点
        self.subgraph_qkc_data=self.get_subgraph("qkc")
        self.subgraph_qs_data=self.get_subgraph("qs")

        self.g=dgl.heterograph(self.graph_data).to(device)
        self.set_feature(mode="graph",hasgrad=False)
        
        if preG is True:
            self.g_qkc=dgl.heterograph(self.subgraph_qkc_data).to(device)
            self.g_qs=dgl.heterograph(self.subgraph_qs_data).to(device)
            self.set_feature(mode="qkc",hasgrad=True)
            self.set_feature(mode="qs",hasgrad=True)

        self.numN_q=self.g.num_nodes('question')
        self.numN_kc=self.g.num_nodes('concept')
        self.numN_stu=self.g.num_nodes('student')
        self.numE_qkc=self.g.num_edges('contain')
        self.numE_kcs=self.g.num_edges('dependOn')
        self.numE_sq=self.g.num_edges('finish')

        self.show_info()

        self.GCN=HeteroGCN(self.g,[self.get_subgraph("qkc"),self.get_subgraph("qs")],self.numN_q,self.numN_kc,self.numN_stu,d_model)

        self.layer_norm_layer0=nn.LayerNorm(d_model)
        self.layer_norm_layer1=nn.LayerNorm(d_model)

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

    def show_info(self):
        print(self.g.ntypes)
        print(self.g.etypes)
        print(self.g.canonical_etypes)
        print(self.g)
        print(self.g.metagraph().edges())
        print(self.g.num_nodes())
        print(self.g.num_nodes('question'))
        print(self.g.nodes('student'))   

        # options = {
        #     'node_color': 'black',
        #     'node_size': 20,
        #     'width': 1,
        # }
        # # 使用draw()函数绘制图形
        # plt.figure(figsize=[15,7])
        # nx.draw(dgl.to_networkx(dgl.to_homogeneous(self.g.to("cpu"))), **options)
        # plt.savefig("graph.png")

    def get_subgraph(self,mode="qkc"):
        if mode=="qkc":
            subgraph_data={
                ('question','contain','concept'):(torch.cat((self.M_qkc.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0),torch.cat((self.M_qkc.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0)),
                ('concept','dependOn','concept'):(torch.cat((self.M_kcs.indices()[0],torch.tensor([cg.NUM_KC]).to(device)),dim=0),torch.cat((self.M_kcs.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0)),
                ('concept','contain-ed','question'):(torch.cat((self.M_qkc.indices()[1],torch.tensor([cg.NUM_KC]).to(device)),dim=0),torch.cat((self.M_qkc.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0))
            }
        if mode=='qs':
            subgraph_data={
                ('student','finish','question'):(torch.cat((self.M_qs.indices()[1],torch.tensor([cg.NUM_STU]).to(device)),dim=0),torch.cat((self.M_qs.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0)),
                ('question','finish-ed','student'):(torch.cat((self.M_qs.indices()[0],torch.tensor([cg.NUM_Q]).to(device)),dim=0),torch.cat((self.M_qs.indices()[1],torch.tensor([cg.NUM_STU]).to(device)),dim=0))
            }
        return subgraph_data
    
    def forward(self,x_question,y_knowledge,seq_q):
        new_feature=self.Temporal2Spatial(x_question,y_knowledge)
        new_feature=self.layer_norm_layer0(new_feature)

        h_qkc,h_qs,seq_qq=self.GCN(new_feature,seq_q)
        hq_qkc=h_qkc['question']
        hq_qs=h_qs['question']

        hq_qkc,hq_qs=self.Spatial2Temproal(hq_qkc,hq_qs,seq_qq)

        return hq_qkc,hq_qs #[numq,bs,dmodel*2]
    

    def Temporal2Spatial(self,x_question,y_knowledge):
        return x_question

    def Spatial2Temproal(self,hq_qkc,hq_qs,seq_qq):
        #hq_qkc:[numq,bs,dmodel*2]
        #seq_q:[bs,seqlen]
        numq,bs,dmodel2=hq_qkc.shape
        hq_qkc=hq_qkc.permute(1,0,2).reshape(numq*bs,-1)[seq_qq].reshape(bs,-1,dmodel2) #[bs,seqlen,dmodel*2]
        hq_qs=hq_qs.permute(1,0,2).reshape(numq*bs,-1)[seq_qq].reshape(bs,-1,dmodel2)
        # print(hq_qkc.shape)
        return hq_qkc,hq_qs
