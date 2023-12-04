Knowledge tracking code
Integrating Global Information and Hidden Information for Knowledge Tracing

Requirements
PyTorch==1.7.0
Python==3.8.0
Running
We evaluate our method on four datasets including Statics2011, ASSISTments2009, ASSISTments2015 and ASSISTments2017.

     

for different dataset
Statics2011
python main.py --dataset 'statics'
ASSISTments2009
python main.py --dataset 'assist2009_pid'
ASSISTments2015
python main.py --dataset 'assist2015'
ASSISTments2017
python main.py --dataset 'assist2017_pid'
Evaluated results (AUC scores) will be saved in statics_test_result.txt, assist2009_pid_test_result.txt, assist2015_test_result.txt, and assist2017_pid_test_result.txt, respectively.