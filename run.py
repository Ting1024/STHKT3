
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import MyDataset
import config as cg
import numpy as np
import torch.optim as optim
from sklearn.metrics import roc_auc_score,accuracy_score
from hgakt import HGAKT
import matplotlib.pyplot as plt
import os

device=cg.DEVICE
CUDA_LAUNCH_BLOCKING=1

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def masking(preds,targets):
    mask=torch.where(targets!=2,torch.tensor([1]).to(device),targets)
    mask=torch.where(mask==2,torch.tensor([0]).to(device),mask)
    mask=mask.to(torch.bool)
    mask_preds=torch.masked_select(preds,mask)
    mask_targets=torch.masked_select(targets,mask)
    return mask_preds,mask_targets

def mask_loss(seq_q,preds,targets):
    mask=torch.where(seq_q!=cg.NUM_Q,torch.tensor([1]).to(device),seq_q)
    mask=torch.where(mask==cg.NUM_Q,torch.tensor([0]).to(device),mask)
    mask=mask.to(torch.bool)
    mask_preds=torch.masked_select(preds,mask)
    valid_num=len(mask_preds)
    mask_targets=torch.flatten(targets)[:valid_num]
    loss_fn = nn.BCELoss()
    loss=loss_fn(mask_preds.to(torch.float),mask_targets.to(torch.float))
    return loss

def tensor_to_numpy(tensor,device=cg.DEVICE):
    if device=="cpu":
        return tensor.numpy()
    tensor_on_cpu = tensor.cpu()
    tensor=tensor_on_cpu.numpy()
    return tensor

def mask_AUC(preds,targets): #用ancs替换，内存会变小
    mask_preds,mask_targets=masking(preds,targets)
    auc=roc_auc_score(tensor_to_numpy(mask_targets), tensor_to_numpy(mask_preds))
    return auc


def train(model):
    if not os.path.exists(cg.MODEL_SAVE_FOLDER):
        os.makedirs(cg.MODEL_SAVE_FOLDER)

    dataset=MyDataset(cg.TRAIN_FILE,cg.MAX_SEQ,cg.MIN_SEQ)
    dataset_test=MyDataset(cg.TEST_FILE,cg.MAX_SEQ,cg.MIN_SEQ)
    dataloader=DataLoader(dataset,batch_size=cg.BATCH_SIZE)

    loss_fn = nn.BCELoss()
    optimizer = NoamOpt(cg.D_MODEL, 1, 4000 ,optim.Adam(model.parameters(), lr=cg.LR))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer.optimizer, gamma=0.5)

    losses_train=[]
    losses_test=[]
    losses_train_sum=[]

    bestauc=0

    for epoch in range(cg.MAX_EPOCH):
        print("\n\n\n",epoch,"\n\n\n")
        

        for users,problems,ansCs,concepts in dataloader:
            optimizer.optimizer.zero_grad()
            
            users=users.to(device)
            problems=problems.to(device)
            ansCs=ansCs.to(device)
            concepts=concepts.to(device)

            model.train()
            
            preds=model(problems,ansCs) 

            mask_preds,mask_targets=masking(preds.squeeze(-1),ansCs)
            loss=loss_fn(mask_preds.to(torch.float),mask_targets.to(torch.float))

            losses_train.append(loss.item())
            # print(loss.item())

            # if loss.item()<0.2:
            # torch.save(model.state_dict(),cg.SAVEMODEL)
            # break

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            bestauc = test(model,bestauc,dataset_test)
            # losses_test.append(loss_test)

        loss_item=sum(losses_train)/len(losses_train)
        losses_train_sum.append(loss_item)

        # torch.save(model.state_dict(),cg.SAVEMODEL)
        print(losses_train_sum)

        # if loss_item<0.609:
            # torch.save(model.state_dict(),cg.SAVEMODEL)
            # break
        # ploting(losses_train_sum,cg.PIC_SAVE_FOLDER+"loss_train22_sum.png")
        
        

def ploting(losses,saving_path,xlabel="Epoch",ylabel="Loss"):
    print("Drawing……")
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(saving_path, dpi=300)



def test(model,bestauc=1,dataset_test=None):
    # dataset_test=MyDataset(cg.TEST_FILE,cg.MAX_SEQ,cg.MIN_SEQ)
    dataloader_test=DataLoader(dataset_test,batch_size=cg.BATCH_SIZE)
    loss_fn = nn.BCELoss()
    model.eval()

    auc=0
    total_pred=[]
    total_ansCs=[]

    for users,problems,ansCs,concepts in dataloader_test:
        users=users.to(device)
        problems=problems.to(device)
        ansCs=ansCs.to(device)
        # ansCs=torch.randint(0,2,ansCs.shape).to(device) # 验证结果：不是训练集测试集划分问题
        concepts=concepts.to(device)
        # ansCs=torch.zeros(ansCs.shape).to(device)
        # ansCs=torch.where(concepts==123,torch.tensor([2]).to(device),ansCs)

        with torch.no_grad():
            preds=model(problems,ansCs) #problems替换为concepts
            mask_preds,mask_targets=masking(preds.squeeze(-1),ansCs)
            mask_preds=mask_preds.reshape(-1)
            mask_targets=mask_targets.reshape(-1)

        total_pred.extend(mask_preds.tolist())
        total_ansCs.extend(mask_targets.tolist())
    acc=accuracy_score(np.array(total_ansCs), np.where(np.array(total_pred) > 0.5, 1, 0))
    auc=roc_auc_score(np.array(total_ansCs), np.array(total_pred))
    if auc > bestauc:
        bestauc=auc
    print("test_acc=",acc)
    print("=====================test_auc=",auc,"==========================")
    print("=====================bestauc=",bestauc,"==========================")
        
    # loss=loss_fn(mask_preds.to(torch.float),mask_targets.to(torch.float))

    model.train()
    return bestauc


if __name__=="__main__":
    model=HGAKT(cg.D_MODEL,cg.LR,preG=cg.HAS_PREG,hawkes=cg.HAS_HAWKES).to(device)
    train(model)
