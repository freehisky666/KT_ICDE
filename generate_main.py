# -*- coding: utf-8 -*-
import os
import os.path
import math
import gc
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from model import KT_backbone
from dataset import DATA, PID_DATA
from sklearn.metrics import roc_auc_score
from utils import KTLoss, _l2_normalize_adv
from pytorchtools import EarlyStopping
import torch.nn as nn
import copy
import constants as C
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
 
        self.fc1 = nn.Linear(C.Max_step, C.Max_step)
        self.fc2 = nn.Linear(C.Max_step, int(C.Max_step/1.5))
        self.fc3 = nn.Linear(int(C.Max_step/1.5), int(C.Max_step/2))
        self.fc41 = nn.Linear(int(C.Max_step/2), int(C.Max_step/4))
        self.fc42 = nn.Linear(int(C.Max_step/2), int(C.Max_step/4))
        self.fc5 = nn.Linear(int(C.Max_step/4), int(C.Max_step/2))
        self.fc6 = nn.Linear(int(C.Max_step/2), int(C.Max_step/1.5))
        self.fc7 = nn.Linear(int(C.Max_step/1.5), C.Max_step)
        self.fc8 = nn.Linear(C.Max_step, C.Max_step)
 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.sigmoid(self.fc2(h1))
        h3 = self.fc3(h2)
        return self.fc41(h3), self.fc42(h3)
 
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
 
    def decode(self, z):
        h4 = self.fc5(z)
        h5 = self.sigmoid(self.fc6(h4))
        h6 = self.fc7(h5)
        return self.fc8(h6)
 
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, C.Max_step))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

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
    
def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x.view(-1, C.Max_step))
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
 
    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))
 
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
 
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
    

parser = argparse.ArgumentParser(description='Script to train KT')
parser.add_argument('--max_iter', type=int, default=150, help='number of iterations')
parser.add_argument('--seed', type=int, default=224, help='default seed')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--lr-decay', type=int, default=50, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--dataset', type=str, default=C.DATASET)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--epsilon', type=float, default=10)
params = parser.parse_args()
dataset = params.dataset

if dataset in {"statics"}:
    params.skill_emb_dim = 512
    params.answer_emb_dim = 60
    params.hidden_emb_dim = 96
    params.n_skill = 1223
    params.batch_size = 24
    params.seqlen = 200
    params.data_dir = 'dataset/'+dataset
    params.data_name = dataset
    params.beta = 0.2
    params.epsilon=12
    params.weight_decay=1.5e-4

if dataset in {"assist2009_pid"}:
    params.skill_emb_dim = 256
    params.answer_emb_dim = 80
    params.hidden_emb_dim = 160
    params.n_skill = 110
    params.batch_size = 24
    params.seqlen = 200
    params.data_dir = 'dataset/'+dataset
    params.data_name = dataset
    params.beta = 0.2
    params.epsilon=10
    params.weight_decay=5e-5
    
if dataset in {"assist2015"}:
    params.skill_emb_dim = 30
    params.answer_emb_dim = 30
    params.hidden_emb_dim = 80
    params.n_skill = 100
    params.batch_size = 24
    params.seqlen = 200
    params.data_dir = 'dataset/'+dataset
    params.data_name = dataset
    params.beta = 1
    params.epsilon=15
    params.weight_decay=0
    
if dataset in {"assist2017_pid"}:
    params.skill_emb_dim = 256
    params.answer_emb_dim = 60
    params.hidden_emb_dim = 160
    params.n_skill = 102
    params.batch_size = 24
    params.seqlen = 200
    params.data_dir = 'dataset/'+dataset
    params.data_name = dataset
    params.beta = 1
    params.epsilon=12
    params.weight_decay=1e-5



for dataset_set_index in range(1,6):
    params.dataset_set_index=dataset_set_index
    
    # dataset
    if "pid" not in params.data_name:
        dat = DATA(n_question=params.n_skill,
                    seqlen=params.seqlen, separate_char=',', maxstep=C.Max_step)
    else:
        dat = PID_DATA(n_question=params.n_skill,
                        seqlen=params.seqlen, separate_char=',', maxstep=C.Max_step)

    train_data_path = params.data_dir + "/" + \
        params.data_name + "_train"+str(params.dataset_set_index)+".csv"
    valid_data_path = params.data_dir + "/" + \
        params.data_name + "_valid"+str(params.dataset_set_index)+".csv"
    test_data_path = params.data_dir + "/" + \
        params.data_name + "_test"+str(params.dataset_set_index)+".csv"

    train_skill_data, train_answer_data = dat.load_data(train_data_path)
    val_skill_data, val_answer_data = dat.load_data(valid_data_path)
    test_skill_data,  test_answer_data = dat.load_data(test_data_path)
    train_loader=torch.FloatTensor(train_skill_data)
    train_loader = torch.utils.data.DataLoader(train_loader,batch_size=params.batch_size, shuffle=True)
    test_loader=torch.FloatTensor(test_skill_data)
    test_loader = torch.utils.data.DataLoader(test_loader,batch_size=params.batch_size, shuffle=True)
    model = VAE()
    reconstruction_function = nn.MSELoss()
    reconstruction_function.size_average = False
    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0)

    epochs=C.generate_epochs
    for epoch in range(1, epochs):
        train(epoch)
        test(epoch)
    model_state = copy.deepcopy(model.state_dict())
    save_model_file = os.path.join('./fold{}'.format(params.dataset_set_index))
    if not os.path.exists(save_model_file):
        os.mkdir(save_model_file)
    model_save_path=os.path.join(save_model_file, 'model_generate0.pt')
    torch.save(model_state, model_save_path)
    loaded_paras = torch.load(model_save_path)


    model2=VAE2()
    model_state2 = copy.deepcopy(model2.state_dict())
    for key in model_state2:
        if key in loaded_paras and model_state2[key].size() == loaded_paras[key].size():
            print("Successful loading parameter:", key)
            model_state2[key] = loaded_paras[key]

    
    model_save_path1=os.path.join(save_model_file, 'model_generate1.pt')
    torch.save(model_state2, model_save_path1)
    loaded_paras1 = torch.load(model_save_path1)

