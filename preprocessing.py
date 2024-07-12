import os
import pandas as pd
import glob
import pandas as pd
import torch
import pickle
import math
import config as cg
from tqdm import tqdm
import re
import json

device=cg.DEVICE

class NormData(object):
    def __init__(self,dataset="Assist2009",hasNorm=False,cut_direction="vertical",proportion=0.8):
        print("###################init Norming###################")

        if dataset=="Assist2009":            
            filename=cg.DATASET_ROOTPATH+"normdata/skill_builder_data.csv" #r
            self.trainfile=cg.TRAIN_FILE #w
            self.testfile=cg.TEST_FILE #w

            self.df=pd.read_csv(filename)[['order_id','user_id','problem_id','correct','skill_id','ms_first_response']]
            self.df=self.df.dropna().reset_index(drop=True) #去掉缺失值之后需要重新编号索引
        elif dataset=="Assistment2017":
            filename=cg.DATASET_ROOTPATH+"sampledata2.csv"
            self.trainfile=cg.TRAIN_FILE #w
            self.testfile=cg.TEST_FILE #w

            self.df=pd.read_csv(filename)[['order_id','user_id','problem_id','correct','skill_id','ms_first_response']]
            self.df=self.df.dropna().reset_index(drop=True) #去掉缺失值之后需要重新编号索引
        elif dataset=="Stat2011":
            filename=cg.DATASET_ROOTPATH+"all_data.csv"
            self.trainfile=cg.TRAIN_FILE #w
            self.testfile=cg.TEST_FILE #w

            self.df=pd.read_csv(filename)[['order_id','user_id','problem_id','correct','skill_id','ms_first_response']]
            self.df=self.df.dropna().reset_index(drop=True) #去掉缺失值之后需要重新编号索引
        elif dataset=="Aaai23":
            self.trainfile=cg.TRAIN_FILE #w
            self.testfile=cg.TEST_FILE #w
            
            self.df_train=pd.read_csv("/data/Ting123/Aaai23/train_valid_sequences.csv")
            self.df_test=pd.read_csv("/data/Ting123/Aaai23/pykt_test.csv")
            self.df=pd.concat((self.df_train,self.df_test),axis=0)
            self.df=self.df.dropna().reset_index(drop=True) 


        if hasNorm is True:
            self.Norm(dataset)

        if dataset!='Aaai23':
            if cut_direction=="vertical":
                num_data=len(self.df)
                split_point=math.floor(num_data*proportion)
                self.df_test=self.df[split_point:]
                self.df_train=self.df[:split_point]
            else:
                print("水平方向切割数据集还未开发")

        self.df_train.to_csv(self.trainfile)
        self.df_test.to_csv(self.testfile)

    def Norm(self,dataset):
        if dataset!='Aaai23':
            print("#####################Norming#####################")

            q2index={}
            index2q={}
            kc2index={}
            index2kc={}
            s2index={}
            index2s={}

            with tqdm(total=len(self.df)) as pbar:
                pbar.set_description('Norming:')

                for i,row in self.df.iterrows():
                    q=row['problem_id']
                    if q in q2index.keys():
                        index=q2index[q] #获得原始q对应的index
                    else:
                        index=len(q2index)
                        q2index[q]=index
                        index2q[index]=q
                    self.df.iloc[i,2]=index #problem_id在['order_id','user_id','problem_id','correct','skill_id','ms_first_response']中排第三列            

                    kc=row['skill_id']
                    if kc in kc2index.keys():
                        index=kc2index[kc]
                    else:
                        index=len(kc2index)
                        kc2index[kc]=index
                        index2kc[index]=kc
                    self.df.iloc[i,4]=index

                    s=row['user_id']
                    if s in s2index.keys():
                        index=s2index[s]
                    else:
                        index=len(s2index)
                        s2index[s]=index
                        index2s[index]=s
                    self.df.iloc[i,1]=index

                    pbar.update(1)
                    
        else:
            print("#####################Norming2#####################")
            q2index={q:q for q in range(cg.NUM_Q)}
            index2q={q:q for q in range(cg.NUM_Q)}
            kc2index={kc:kc for kc in range(cg.NUM_KC)}
            index2kc={kc:kc for kc in range(cg.NUM_KC)}
            s2index={s:s for s in range (cg.NUM_STU)}
            index2s={s:s for s in range (cg.NUM_STU)}

            
        with open(cg.DICT_Q2INDEX,'wb') as f:
            pickle.dump(q2index, f)
        f.close()
        with open(cg.DICT_INDEX2Q,'wb') as f:
            pickle.dump(index2q, f)
        f.close()
        with open(cg.DICT_KC2INDEX,'wb') as f:
            pickle.dump(kc2index, f)
        f.close()
        with open(cg.DICT_INDEX2KC,'wb') as f:
            pickle.dump(index2kc, f)
        f.close()
        with open(cg.DICT_STU2,'wb') as f:
            pickle.dump(q2index, f)
        f.close()
        with open(cg.DICT_INDEX2Q,'wb') as f:
            pickle.dump(index2q, f)
        f.close()

            


class preprocess(object):
    def __init__(self,re_matrix=False,mode=None):
        print("###################init preprocee###################")
        trainfile=cg.TRAIN_FILE
        testfile=cg.TEST_FILE

        self.df_train=pd.read_csv(trainfile)[['order_id','user_id','problem_id','correct','skill_id','ms_first_response']]
        self.df_test=pd.read_csv(testfile)[['order_id','user_id','problem_id','correct','skill_id','ms_first_response']]
        print("Finish loading csv ")
        self.df=pd.concat([self.df_train, self.df_test], axis=0)

        self.num_stu,self.num_q,self.num_kc=self.get_all_info()

        if re_matrix is True:
            if mode is None:
                torch.save(self.get_M_kcs(),cg.MATRIX_KCS)
                torch.save(self.get_M_qkc(),cg.MATRIX_QKC)
                torch.save(self.get_M_qs(),cg.MATRIX_QS)
            elif mode=="qkc":
                torch.save(self.get_M_qkc(),cg.MATRIX_QKC)
            elif mode=="qs":
                torch.save(self.get_M_qs(),cg.MATRIX_QS)


    def get_all_info(self):
        stu_num=self.df['user_id'].nunique()
        q_num=self.df['problem_id'].nunique()
        kc_num=self.df['skill_id'].nunique()
        print(stu_num,q_num,kc_num)
        return stu_num,q_num,kc_num

    
    def get_M_kcs(self):
        #self.num_kc=123
        i=torch.stack((torch.arange(self.num_kc),torch.arange(self.num_kc)),dim=0).type(torch.LongTensor)
        v=torch.ones(self.num_kc).type(torch.FloatTensor)
        s=torch.Size([self.num_kc,self.num_kc])
        M_kcs=torch.sparse.FloatTensor(i,v,s)
        return M_kcs


    def get_M_qkc(self):
        
        q_list=[]
        kc_list=[]
        qkc_list=[]

        with tqdm(total=len(self.df)) as pbar:
            pbar.set_description('Mqkc:')

            for i,row in self.df.iterrows():
                # print(i)
                q=row['problem_id']
                kc=row['skill_id'] 
                # if (q,kc) in qkc_list:
                #     continue
                # q_list.append(q)
                # kc_list.append(kc)
                qkc_list.append((q,kc))

                pbar.update(1)

        kc_list=list(set(kc_list))
        q_list = [qkc[0] for qkc in qkc_list]
        kc_list=[qkc[1] for qkc in qkc_list]

        i = torch.stack((torch.tensor(q_list), torch.tensor(kc_list)), dim=0).type(torch.LongTensor)
        v = torch.ones(len(qkc_list)).type(torch.FloatTensor)
        s = torch.Size([self.num_q,self.num_kc]) #17751,123

        result = torch.sparse.FloatTensor(i,v,s)
        return result


    def get_M_qs(self):
        q_list=[]
        s_list=[]
        qs_list=[]

        with tqdm(total=len(self.df)) as pbar:
            pbar.set_description('Mqs:')

            for i,row in self.df.iterrows():
                # print(i)
                q=row['problem_id']
                s=row['user_id']
                # if (q,s) in qs_list:
                #     continue
                # q_list.append(q)
                # s_list.append(s)
                qs_list.append((q,s))

                pbar.update(1)
        
        qs_list=list(set(qs_list))
        q_list = [qs[0] for qs in qs_list]
        s_list=[qs[1] for qs in qs_list]

        i = torch.stack((torch.tensor(q_list), torch.tensor(s_list)), dim=0).type(torch.LongTensor)
        v = torch.ones(len(qs_list)).type(torch.FloatTensor)
        s = torch.Size([self.num_q,self.num_stu]) #17751,4613

        result = torch.sparse.FloatTensor(i,v,s)
        return result
    

class preprocess2(object):
    def __init__(self,re_matrix=False,mode=None):
        print("###################init preprocee2###################")
        trainfile=cg.TRAIN_FILE
        testfile=cg.TEST_FILE
        self.df_train=pd.read_csv(trainfile)[['uid','questions','responses','concepts']]
        self.df_test=pd.read_csv(testfile)[['uid','questions','responses','concepts']]
        print("Finish loading csv ")
        self.df=pd.concat([self.df_train, self.df_test], axis=0)
        
        self.num_stu,self.num_q,self.num_kc=cg.NUM_STU,cg.NUM_Q,cg.NUM_KC

        # if re_matrix is True:
        #     if mode is None:
        #         torch.save(self.get_M_kcs(),cg.MATRIX_KCS)
        #         torch.save(self.get_M_qkc(),cg.MATRIX_QKC)
        #         torch.save(self.get_M_qs(),cg.MATRIX_QS)
        #     elif mode=="qkc":
        #         torch.save(self.get_M_qkc(),cg.MATRIX_QKC)
        #     elif mode=="qs":
        #         torch.save(self.get_M_qs(),cg.MATRIX_QS)
        M_kcs,M_qkc=self.get_M_qkcs()
        torch.save(M_kcs,cg.MATRIX_KCS)
        torch.save(M_qkc,cg.MATRIX_QKC)
                
    def extract_kcs(self,strings,kc2index_dict):
        result = []
        numbers =[]
        # 提取数字并添加到嵌套列表中
        for string in strings:
            digits = [kc2index_dict[num] for num in re.findall(r'\d+', string)]
            print(digits)
            digits[0]=digits[0][1:]
            result.append(digits)
            digits = re.findall(r'\d+', string)
            digits=[kc2index_dict[num] for num in digits]
            digits=digits[1:]
            numbers.extend(digits)
            numbers=list(set(numbers))
        return result,numbers

                
    def get_M_qkcs(self):
        kcs_list=[]
        qkc_list=[]
        with open(cg.DATASET_ROOTPATH+"keyid2idx.json",'r') as f:
            datas=json.load(f)
            q2index_dict=dict(datas['questions'])
            kc2index_dict=dict(datas['concepts'])
            # s2index_dict=dict(datas['students'])
        # #self.num_kc=123
        # i=torch.stack((torch.arange(self.num_kc),torch.arange(self.num_kc)),dim=0).type(torch.LongTensor)
        # v=torch.ones(self.num_kc).type(torch.FloatTensor)
        # s=torch.Size([self.num_kc,self.num_kc])
        # M_kcs=torch.sparse.FloatTensor(i,v,s)
        # return M_kcs
            with open(cg.DATASET_ROOTPATH+"questions.json", 'r') as f:
                datas2 = json.load(f)
                for key,data in tqdm(datas2.items(),'qkc&kcs'):
                    question=q2index_dict[key]
                    concept_routes_lists,concepts=self.extract_kcs(data['concept_routes'],kc2index_dict)
                    for concept_route in concept_routes_lists:
                        pairs = [(numbers[i], numbers[i+1]) for i in range(len(numbers)-1)]
                        kcs_list.extend(pairs)
                    qkc_list.extend([(question,num) for num in concepts])
                    
        kcs_list=list(set(kcs_list))
        qkc_list=list(set(qkc_list))
        kc0_list=[kc[0] for kc in kcs_list]
        kc1_list=[kc[1] for kc in kcs_list]
        q_list = [qkc[0] for qkc in qkc_list]
        kc_list=[qkc[1] for qkc in qkc_list]
                                    
        i = torch.stack((torch.tensor(kc0_list), torch.tensor(kc1_list)), dim=0).type(torch.LongTensor)
        v = torch.ones(len(kcs_list)).type(torch.FloatTensor)
        s = torch.Size([self.num_kc,self.num_kc]) 
        M_kcs = torch.sparse.FloatTensor(i,v,s)
                                    
        i = torch.stack((torch.tensor(q_list), torch.tensor(kc_list)), dim=0).type(torch.LongTensor)
        v = torch.ones(len(qkc_list)).type(torch.FloatTensor)
        s = torch.Size([self.num_q,self.num_kc]) #17751,123
        M_qkc = torch.sparse.FloatTensor(i,v,s)  
                                    
        return M_kcs,M_qkc
   


if __name__=="__main__":
    # pass
    # norm=NormData(dataset=cg.DATASET,hasNorm=True)
    pre=preprocess2(re_matrix=True)
    
