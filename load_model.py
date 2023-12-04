from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import copy
import os
from PIL import Image
#from pylab import *
import numpy as np
from model import KT_backbone
import math
import constants as C
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VAE2(nn.Module):
    def __init__(self):
        super(VAE2, self).__init__()
        self.fc5 = nn.Linear(int(C.Max_step/4), int(C.Max_step/2))
        self.fc6 = nn.Linear(int(C.Max_step/2), int(C.Max_step/1.5))
        self.fc7 = nn.Linear(int(C.Max_step/1.5), C.Max_step)
        self.fc8 = nn.Linear(C.Max_step, C.Max_step)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self, z):
        h4 = self.fc5(z)
        h5 = self.sigmoid(self.fc6(h4))
        h6 = self.fc7(h5)
        return self.fc8(h6)
 
    def forward(self, x):
        z = self.decode(x)
        return z
def generate(n_ge):

    model2=VAE2()
    model_save_path1 = os.path.join('model_generate1.pt')
    loaded_paras1 = torch.load(model_save_path1)
    model2.load_state_dict(loaded_paras1)
    model2.eval()
    u=torch.zeros(n_ge,int(C.Max_step/4))
    y=torch.ones(n_ge,int(C.Max_step/4))
    a=torch.randn(n_ge,int(C.Max_step/4))
    with torch.no_grad():
        std = y.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        output = eps.mul(std).add_(u)
        pic_dim=model2(output)
    pic_dim=np.array(pic_dim)
    pic_dim_1=pic_dim[:,0:C.Max_step]
    pic_dim_2=pic_dim[:,C.Max_step:C.Max_step*2]
    return pic_dim_1,pic_dim_2

def generate2(n_ge):
    y=torch.ones(n_ge,C.Max_step)
    u=torch.zeros(n_ge,C.Max_step)
    return y,u

def generate_skill(n_ge,dataset_set_index):

    y2=np.ones(n_ge*C.Max_step)
    y2=y2.reshape(n_ge,C.Max_step)
    model2=VAE2()
    model_save_path1 = os.path.join('./fold{}'.format(dataset_set_index))
    model_save_path1 = os.path.join(model_save_path1,'model_generate1.pt')
    loaded_paras1 = torch.load(model_save_path1)
    model2.load_state_dict(loaded_paras1)
    model2.eval()
    u=torch.zeros(n_ge,int(C.Max_step/4))
    y=torch.ones(n_ge,int(C.Max_step/4))
    a=torch.randn(n_ge,int(C.Max_step/4))
    with torch.no_grad():
        std = y.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        output = eps.mul(std).add_(u)
        pic_dim=model2(output)
    pic_dim=np.array(pic_dim)
    return pic_dim

def generate_answer(n_ge,generate_skill,dataset_set_index):
    save_model_file = os.path.join('./fold{}'.format(dataset_set_index))
    load_model_path=os.path.join(save_model_file, 'pre_train_kt_model_best.pt')
    net = KT_backbone(C.Num_of_skill_emb_dim, C.Num_of_answer_emb_dim, C.Num_of_hidden_emb_dim, C.NUM_OF_QUESTIONS)
    net.load_state_dict(torch.load(load_model_path))
    net.to(device)
    y2=np.ones(n_ge*C.Max_step)
    ansewer_allone=y2.reshape(n_ge,C.Max_step)
    net.eval()
    with torch.no_grad():
        skill=torch.LongTensor(generate_skill)
        answer=torch.LongTensor(ansewer_allone)
        skill = torch.where(skill==-1, torch.tensor([C.NUM_OF_QUESTIONS]), skill)
        answer = torch.where(answer==-1, torch.tensor([2]), answer)
        
        for i in range(0,len(skill)):
            for j in range(0,len(skill[0])):
                if skill[i][j]==C.NUM_OF_QUESTIONS or skill[i][j]<1:
                    skill[i,j:]=C.NUM_OF_QUESTIONS
                    break
        skill,answer=skill.to(device),answer.to(device)
        pred_res, _ = net(skill, answer)
        
    return pred_res

def generate_answer2(generate_skill,dataset_set_index):
    save_model_file = os.path.join('./fold{}'.format(dataset_set_index))
    load_model_path=os.path.join(save_model_file, 'pre_train_kt_model_best.pt')
    net = KT_backbone(C.Num_of_skill_emb_dim, C.Num_of_answer_emb_dim, C.Num_of_hidden_emb_dim, C.NUM_OF_QUESTIONS)
    net.load_state_dict(torch.load(load_model_path))
    net.to(device)
    final_pred=torch.randn(0,C.Max_step)
    final_pred=final_pred.to(device)
    net.eval()
    genetate_N = int(math.ceil(len(generate_skill) / 36))
    for idx in range(genetate_N):
        generate_batch   = generate_skill[idx*36:(idx+1)*36]
        with torch.no_grad():
            skill=torch.LongTensor(generate_batch)
            y2=np.ones(len(generate_batch)*C.Max_step)
            ansewer_allone=y2.reshape(len(generate_batch),C.Max_step)
            answer=torch.LongTensor(ansewer_allone)
            skill = torch.where(skill==-1, torch.tensor([C.NUM_OF_QUESTIONS]), skill)
            answer = torch.where(answer==-1, torch.tensor([2]), answer)

            for i in range(0,len(skill)):
                for j in range(0,len(skill[0])):
                    if skill[i][j]==C.NUM_OF_QUESTIONS or skill[i][j]<1:
                        skill[i,j:]=C.NUM_OF_QUESTIONS
                        break
            skill,answer=skill.to(device),answer.to(device)
            pred_res, _ = net(skill, answer)
            temp_train_answer_last1=pred_res[:,-1]
            temp_train_answer_last1=temp_train_answer_last1.view(len(temp_train_answer_last1),-1)
            pred_res=torch.cat([pred_res,temp_train_answer_last1],1)
        final_pred=torch.cat([final_pred,pred_res],axis=0)
    return final_pred

def skill_transform(skill):
    temp_id=[]
    temp_count=[]
    for i in range (skill.shape[0]):
        for j in range (skill.shape[1]):
            if j ==0:
                temp_id.append([])
                temp_count.append([])
                temp_id[i].append(skill[i][j])
                temp_count[i].append(1)
            elif j !=0 and skill[i][j] != skill[i][j-1]:
                temp_id[i].append(skill[i][j])
                temp_count[i].append(1)
            elif j !=0 and skill[i][j] == skill[i][j-1]:
                temp_count[i][-1]=temp_count[i][-1]+1
    temp_id,temp_count=np.array(temp_id,dtype=object),np.array(temp_count,dtype=object)
    return temp_id,temp_count

def answer_transform(skill_count):
    temp_id=[]
    for i in range(skill_count.shape[0]):
        temp_id.append([])
        for j in range(len(skill_count[i])):
            temp_id[i].append([])
            if j ==0:
                last_label=0
                for m in range(skill_count[i][j]):
                    temp_id[i][j].append(last_label)
                    last_label=last_label+1
            else:
                last_label=temp_id[i][j-1][-1]
                for n in range(skill_count[i][j]):
                    temp_id[i][j].append(last_label+1)
                    last_label=last_label+1
    temp_id=np.array(temp_id,dtype=object)
    return temp_id      


def generate_skill_answer(n_ge,skill_id,skill_count,answer_count,first_answer):
    count_num=[0]*n_ge
    skill=[]
    answer=[]
    for i in range(n_ge):
        skill.append([])
        answer.append([])  
        while count_num[i] < C.Max_step:
            id_rand=random.randint(0,len(skill_id)-1)
            count_rand=random.randint(0,len(skill_id[id_rand])-1)
            for j in range(skill_count[id_rand][count_rand]):
                skill[i].append(skill_id[id_rand][count_rand])
            for m in range(len(answer_count[id_rand][count_rand])):
                answer[i].append(first_answer[id_rand][answer_count[id_rand][count_rand][m]])
            count_num[i]=len(skill[i])
    for i in range(n_ge):
        skill[i]=skill[i][0:C.Max_step]
        answer[i]=answer[i][0:C.Max_step]
    skill,answer=np.array(skill), np.array(answer)
    for n in range(0,n_ge):
        for k in range(0,len(skill[n])):
            if skill[n][k] == -1:
                skill[n][k:C.Max_step]=[-1]*len(skill[k:C.Max_step])
                break
    return skill, answer
# do not use -1 as skill and answer
def generate_skill_answer2(n_ge,skill_id,skill_count,answer_count,first_answer):
    count_num=[0]*n_ge
    skill=[]
    answer=[]
    for pop_id in range(0,len(skill_id)):
        skill_id[pop_id].pop()
        skill_count[pop_id].pop()
    for i in range(n_ge):
        skill.append([])
        answer.append([]) 
        while count_num[i] < C.Max_step:
            id_rand=random.randint(0,len(skill_id)-1)
            count_rand=random.randint(0,len(skill_id[id_rand])-1)
            for j in range(skill_count[id_rand][count_rand]):
                skill[i].append(skill_id[id_rand][count_rand])
            for m in range(len(answer_count[id_rand][count_rand])):
                answer[i].append(first_answer[id_rand][answer_count[id_rand][count_rand][m]])
            count_num[i]=len(skill[i])
    for i in range(n_ge):
        skill[i]=skill[i][0:C.Max_step]
        answer[i]=answer[i][0:C.Max_step]
    skill,answer=np.array(skill), np.array(answer)
    for n in range(0,n_ge):
        slice_id=random.randint(int(C.Max_step)*0.1,C.Max_step-1)
        skill[n][slice_id:C.Max_step]=[-1]*(C.Max_step-slice_id)
        answer[n][slice_id:C.Max_step]=[-1]*(C.Max_step-slice_id)
    return skill, answer

def generate_skill_answer3(n_ge,skill_id,skill_count,answer_count,first_answer):
    count_num=[0]*n_ge
    skill=[]
    answer=[]
    for pop_id in range(0,len(skill_id)):
        skill_id[pop_id].pop()
        skill_count[pop_id].pop()
    for i in range(n_ge):
        skill.append([])
        answer.append([]) 
        while count_num[i] < C.Max_step:
            id_rand=random.randint(0,len(skill_id)-1)
            count_rand=random.randint(0,len(skill_id[id_rand])-1)
            for j in range(skill_count[id_rand][count_rand]):
                skill[i].append(skill_id[id_rand][count_rand])
            for m in range(len(answer_count[id_rand][count_rand])):
                answer[i].append(first_answer[id_rand][answer_count[id_rand][count_rand][m]])
            for len_scare in range(1,4):
                if count_rand+len_scare < len(skill_id[id_rand]) and skill_id[id_rand][count_rand+len_scare] !=-1:
                    for jj in range(skill_count[id_rand][count_rand+len_scare]):
                        skill[i].append(skill_id[id_rand][count_rand+len_scare])
                    for mm in range(len(answer_count[id_rand][count_rand+len_scare])):
                        answer[i].append(first_answer[id_rand][answer_count[id_rand][count_rand+len_scare][mm]])
            count_num[i]=len(skill[i])
    for i in range(n_ge):
        skill[i]=skill[i][0:C.Max_step]
        answer[i]=answer[i][0:C.Max_step]
    #import pdb; pdb.set_trace()    
    skill,answer=np.array(skill), np.array(answer)
    for n in range(0,n_ge):
        slice_id=random.randint(int(C.Max_step)*0.1,C.Max_step-1)
        skill[n][slice_id:C.Max_step]=[-1]*(C.Max_step-slice_id)
        answer[n][slice_id:C.Max_step]=[-1]*(C.Max_step-slice_id)
    return skill, answer

def difficulty_index(skill,answer):
    temp_true={}
    temp_false={}
    for i in range(skill.shape[0]):
        for j in range(skill.shape[1]):
            if skill[i][j] not in temp_true and skill[i][j] !=-1:
                temp_true[skill[i][j]]=0
                temp_false[skill[i][j]]=0
            if answer[i][j]==1:
                temp_true[skill[i][j]]=temp_true[skill[i][j]]+1
            if answer[i][j]==0:
                temp_false[skill[i][j]]=temp_false[skill[i][j]]+1
    output=[]
    #import pdb; pdb.set_trace()    
    for m in range(answer.shape[0]):
        output.append([])
        for n in range(answer.shape[1]):
            if skill[m][n] !=-1:
                output[m].append(temp_true[skill[m][n]]/(temp_true[skill[m][n]]+temp_false[skill[m][n]]))
            if skill[m][n] ==-1:
                output[m].append(-1)
    output=np.array(output)
    condition=(output!=-1)
    output=np.where(condition,(output/0.1).astype(int),output)
    return output 

def confidence_index(data):
    temp_conf=[]
    decay_index=1
    for i in range(data.shape[0]):
        temp_conf.append([])
        if data[i][0]==1:
            temp_conf[i].append(0.7)
        if data[i][0]==0:
            temp_conf[i].append(0.3)
        for j in range(1,data.shape[1]):
            if (data[i][j]==0)&(data[i][j]!=-1):
                temp_conf[i].append((0+temp_conf[i][j-1])/2*decay_index)
            if (data[i][j]==1)&(data[i][j]!=-1):
                temp_conf[i].append((1+temp_conf[i][j-1])/2*decay_index)
            if (data[i][j]==-1):
                temp_conf[i].append(-1)
    temp_conf=np.array(temp_conf)
    condition=(temp_conf!=-1)
    temp_conf=np.where(condition,(temp_conf/0.1).astype(int),temp_conf)
    return temp_conf
