# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants as C
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KT_backbone(nn.Module):
    def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
        super(KT_backbone, self).__init__()
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim*2+self.answer_dim*5, self.hidden_dim, batch_first=True)
        self.rnn2 = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*3, self.output_dim)
        self.sig = nn.Sigmoid()
        ##
        self.fc1 = nn.Linear(self.hidden_dim, self.skill_dim+self.answer_dim)
        self.fc2 = nn.Linear(self.skill_dim+self.answer_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.skill_dim+self.answer_dim+hidden_dim, self.hidden_dim)
        self.relu = nn.functional.relu
        ##
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.confidence_emb=nn.Embedding(13,self.answer_dim)
        self.confidence_emb.weight.data[-1]= 0
        self.difficulty_emb=nn.Embedding(13,self.answer_dim)
        self.difficulty_emb.weight.data[-1]= 0
        
        self.attention_dim = 120
        self.attention_dim_s = 120
        self.attention_dim_b = 120
        self.mlp = nn.Linear(self.skill_dim+self.answer_dim, self.attention_dim)
        #self.mlp = nn.Linear(self.skill_dim+self.answer_dim, self.attention_dim)
        self.mlp2 = nn.Linear(self.answer_dim, self.attention_dim_s)
        self.mlp3 = nn.Linear(self.hidden_dim, self.attention_dim_b)
        self.mlp_co=nn.Linear(20,20)
        self.mlp_di=nn.Linear(20,20)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        self.similarity2 = nn.Linear(self.attention_dim_s, 1, bias=False)
        self.similarity3 = nn.Linear(self.attention_dim_b, 1, bias=False)
        self.similarity_co = nn.Linear(20, 1, bias=False)
        self.similarity_di = nn.Linear(20, 1, bias=False)
        self.dropout=nn.Dropout(C.dropout_pa)
        self.fc_con1=nn.Linear(20,20)
        self.fc_con2=nn.Linear(20,20)
        self.fc_diff1=nn.Linear(20,20)
        self.fc_diff2=nn.Linear(20,20)
        self.fc_ouhe=nn.Linear(C.Max_step,C.Max_step)
        self.fc_couple=nn.Linear(self.output_dim+self.output_dim,self.output_dim+self.output_dim)
        
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.dkq=self.output_dim
        self.dv=self.output_dim
        self.scale=self.dkq**0.5
        self.q_skill=nn.Linear(self.skill_dim,self.dkq)
        self.q_answer=nn.Linear(self.answer_dim,self.dkq)
        
        self.k_skill=nn.Linear(self.skill_dim,self.dkq)
        self.k_answer=nn.Linear(self.answer_dim,self.dkq)
        self.v_skill=nn.Linear(self.skill_dim,self.dv)
        self.v_answer=nn.Linear(self.answer_dim,self.dv)
        
    def attention_normal_skill(self,input):
        q=self.q_skill(input)
        k=self.k_skill(input)
        v=self.v_skill(input)
        att=(q @ k.transpose(-2,-1))*self.scale
        att=att.softmax(dim=-1)    
        output=att @ v
        return output
    def attention_normal_answer(self,input):
        q=self.q_answer(input)
        k=self.k_answer(input)
        v=self.v_answer(input)
        att=(q @ k.transpose(-2,-1))*self.scale
        att=att.softmax(dim=-1)    
        output=att @ v
        return output
        
    def _get_next_pred(self, res, skill):
        #import pdb; pdb.set_trace()
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output
    def attention_module2(self, lstm_output):
        
        att_w = self.mlp2(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity2(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output
    def attention_module3(self, lstm_output):
        
        att_w = self.mlp3(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity3(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output

    def forward(self, skill, answer,confidence,difficulty, perturbation=None):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        confidence_embedding=self.sig(self.confidence_emb(confidence))
        difficulty_embedding=self.sig(self.difficulty_emb(difficulty))
        answer_embedding1=self.relu(answer_embedding+difficulty_embedding+confidence_embedding)
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        skill_answer_embedding=torch.where(answer==0, skill_answer, answer_skill)
        skill_answer_embedding1=skill_answer_embedding
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        out1=self.attention_module2(answer_embedding)
        out2=self.attention_module(skill_answer_embedding)
        out3=torch.cat((out1,out2,answer_embedding1), 2)
        out4,_ = self.rnn(out3)
        out5 = self.attention_module3(out4)
        out6 = torch.cat((out4,out5), 2)
        out6=self.dropout(out6)
        res = self.sig(self.fc(out6))
        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)
        return pred_res, skill_answer_embedding1