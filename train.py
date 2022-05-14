

import pickle as pkl
import torch
from torch.utils import data
import sys
import os
import random

### to be replace are we have our model   ####
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from github_parameters import model_params
import code.classifier_modules as cm 
import code.dataset as dt

def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
###############################################
                                         
class Engine:
    def __init__(self,filename):
        data_path='./data'
        
        hol_tr_dobj_loc = os.path.join(data_path, 'tr_holstep_data.pkl')
        hol_val_dobj_loc = os.path.join(data_path, 'val_holstep_data.pkl')
        hol_te_dobj_loc = os.path.join(data_path, 'te_holstep_data.pkl')
        
        tr_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], model_params['edge_type'])
        te_collator = dt.HolstepCollator(model_params['depth_cap'], model_params['default_pc'], model_params['edge_type'])
        
        
        
        batch_size=128
        shuffle=True
        num_workers=0
        
        self.best_acc=-1
        self.epochs=5
        
        training_set = pkl.load(open(hol_tr_dobj_loc, 'rb'))
        self.training_generator = data.DataLoader(training_set, collate_fn=tr_collator, batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        validation_set = pkl.load(open(hol_val_dobj_loc, 'rb'))
        self.validation_generator = data.DataLoader(validation_set, collate_fn=te_collator, batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        test_set = pkl.load(open(hol_te_dobj_loc, 'rb'))
        self.test_generator = data.DataLoader(test_set, collate_fn=te_collator, batch_size=batch_size,shuffle=False,num_workers=num_workers)
        self.filename=filename
        with open('./'+self.filename+'/log.txt', 'a') as f: 
                print('num_rounds=',model_params['num_rounds'],file=f)
                print('node_emb_dim=',model_params['node_emb_dim'],file=f)
        

        
        
        ###  replace the model with ours ######

        self.model=cm.FormulaRelevanceClassifier(**model_params)

    def train(self):
        for epoch in range(self.epochs):
            with open('./'+self.filename+'/log.txt', 'a') as f: 
                print("\n----------------------------------------new_epoch--------------------------------------\n",file=f)
                print('epoch: ',epoch,file=f)  
            

            self.model.train()
            tic = time.time()
            loss_counter=0
            acc_counter=0
            #with tqdm(total=len(self.training_generator)) as pbar:
            for batch_idx,batch in enumerate(self.training_generator):
                b_loss, b_acc = self.model.train_classifier(batch)
                toc = time.time()
                total_norm=0
                for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2

                
                acc_counter+=b_acc
                loss_counter+=b_loss
                with open('./'+self.filename+'/log.txt', 'a') as f: 
                    print('iteration :',batch_idx,file=f)  
                    print('train acc: ',b_acc,file=f)  
                    print('train loss: ',b_loss,file=f) 
                    print('train gradient: ',total_norm,file=f) 

    
            loss_counter=loss_counter/len(self.training_generator)
            acc_counter=acc_counter/len(self.training_generator)
            
            

            self.model.eval()
            correct=0
            truth=[]
            predict=[]
            with torch.no_grad():
                for batch in self.validation_generator:
                
                    outputs, labels, parse_failures = self.model.run_classifier(batch)
                    
                    pred=torch.ge(outputs,0.5)
                    truth.extend(labels)
                    predict.extend(pred.tolist())


            valacc=accuracy_score(truth,predict)
            cf = confusion_matrix(truth,predict)
            f1score=f1_score(truth,predict)
            if valacc>self.best_acc:
                pkl.dump(self.model, open('./'+self.filename+'/best.pth', 'wb'))
                self.best_acc = valacc
            with open('./'+self.filename+'/log.txt', 'a') as f: 
                print('val acc: ',valacc,file=f)  
                print('val f1: ',f1score,file=f)  
                print('val confusion matrix: ',cf,file=f)  

    def test(self):
        
        self.model = pkl.load(open('./'+self.filename+'/best.pth', 'rb'))
        self.model.eval()

        truth=[]
        predict=[]
        with torch.no_grad():
            for batch in self.test_generator:
                outputs, labels, parse_failures = self.model.run_classifier(batch)
                pred=torch.ge(outputs,0.5)
                truth.extend(labels)
                predict.extend(pred.tolist())


        testacc=accuracy_score(truth,predict)
        cf = confusion_matrix(truth,predict)
        f1score=f1_score(truth,predict)
        with open('./'+self.filename+'/log.txt', 'a') as f: 
            print("\n----------------------------------------test--------------------------------------\n",file=f)
            print('acc: ',testacc,file=f)  
            print('f1: ',f1score,file=f)  
            print('confusion matrix: ',cf,file=f)  

if __name__ == '__main__':   

    paras=sys.argv[1:]
    filename=paras[0]+'_'+paras[1]+'_'+paras[2]
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open('./'+filename+'/log.txt', 'w') as f: 
        print('program start:',file=f)  
    
    seed=int(paras[0])
    
    setseed(seed)
    model_params['num_rounds']= int(paras[1])
    model_params['node_emb_dim']= int(paras[2])
    
    eng =Engine(filename)        
    eng.train()
    eng.test()