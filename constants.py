

datasets = {
    'assist2009_pid' : 'assist2009_pid',
    'assist2015' : 'assist2015',
    'assist2017_pid' : 'assist2017_pid',
    'statics' : 'statics',
    'kddcup2010' : 'kddcup2010',
    'synthetic' : 'synthetic'
}

# question number of each dataset
numbers = {
    'assist2009_pid' : 110,  
    'assist2015' : 100,
    'assist2017_pid' : 102,
    'statics' : 1223, 
    'kddcup2010' : 661,  
    'synthetic' : 50
}

skill_emb_dim={
    'assist2009_pid' : 256,  
    'assist2015' : 30,
    'assist2017_pid' : 256,
    'statics' : 512, 
    'kddcup2010' : 128,  
    'synthetic' : 128
}
answer_emb_dim={
    'assist2009_pid' : 80,  
    'assist2015' : 30,
    'assist2017_pid' : 60,
    'statics' : 60, 
    'kddcup2010' : 64,  
    'synthetic' : 64
}
hidden_emb_dim={
   'assist2009_pid' : 160,  
    'assist2015' : 80,
    'assist2017_pid' : 160,
    'statics' : 96, 
    'kddcup2010' : 128,  
    'synthetic' : 128 
}
avrage_skill_lenth={
    'assist2009_pid' : 2500,  
    'assist2015' : 11904,
    'assist2017_pid' : 1180,
    'statics' : 230, 
    'kddcup2010' : 1000,  
    'synthetic' : 0
}
dropout_pa={
    'assist2009_pid' : 0.5,  
    'assist2015' : 0,
    'assist2017_pid' : 0.5,
    'statics' : 0.9, 
    'kddcup2010' : 0,  
    'synthetic' : 0
}
DATASET = datasets['assist2017_pid']
NUM_OF_QUESTIONS = numbers[DATASET]
Num_of_skill_emb_dim=skill_emb_dim[DATASET]
Num_of_answer_emb_dim=answer_emb_dim[DATASET]
Num_of_hidden_emb_dim=hidden_emb_dim[DATASET]
av_skill_lenth=avrage_skill_lenth[DATASET]
dropout_pa=dropout_pa[DATASET]
Max_step=1000
main_patience=15
pretrain_patience=10
generate_epochs=15