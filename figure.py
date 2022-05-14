import os,sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def t_plot(acc_list,loss_list,gradient_list):
    pass

def plot(acc_list,loss_list,gradient_list):
    epoch = len(acc_list)
    index = np.arange(1,len(acc_list[0])+1,1)
    fig, ax = plt.subplots(epoch,3)
    for i in range(epoch):
        ax[i][0].plot(index,np.array(acc_list[i]),c='b')
        ax[i][0].set_title('epoch %d: acc'%(i+1))
        ax[i][1].plot(index,np.array(loss_list[i]),c='b')
        ax[i][1].set_title('epoch %d: loss'%(i+1))
        ax[i][2].plot(index,np.array(gradient_list[i]),c='b')
        ax[i][2].set_title('epoch %d: gradient'%(i+1))
    fig.tight_layout()
    plt.show()

def plot_2(path,name,acc_list,loss_list,gradient_list):
    index = np.arange(1,len(acc_list)+1,1)
    loss = np.sum(np.array(loss_list),axis=1)/len(loss_list[0])
    gradient = np.sum(np.array(gradient_list),axis=1)/len(gradient_list[0])

    fig, ax = plt.subplots(3,1)
    ax[0].plot(index,np.array(acc_list),c='b')
    ax[0].set_title('val acc')
    ax[0].set_xticks(np.arange(1,6,1))
    ax[1].plot(index,loss,c='b')
    ax[1].set_title('average loss')
    ax[1].set_xticks(np.arange(1,6,1))
    ax[2].plot(index,gradient,c='b')
    ax[2].set_title('average gradient')
    ax[2].set_xticks(np.arange(1,6,1))
    ax[2].set_xlabel('epoch')
    fig.tight_layout()
    plt.suptitle(name,x=0.1)
    plt.savefig(path)
    #plt.show()

def plot_3(path,name,acc_list,loss_list,gradient_list):
    index = np.arange(1,6,1)
    for i in range(len(loss_list)):
        loss_list[i] = np.sum(np.array(loss_list[i]),axis=1)/len(loss_list[i][0])
        gradient_list[i] = np.sum(np.array(gradient_list[i]),axis=1)/len(gradient_list[i][0])
    loss = loss_list
    gradient = gradient_list
    fig, ax = plt.subplots(3,1)

    ax[0].set_title('val acc')
    ax[1].set_title('average loss')
    ax[2].set_title('average gradient')
    ax[0].set_xticks(np.arange(1,6,1))
    ax[1].set_xticks(np.arange(1,6,1))
    ax[2].set_xticks(np.arange(1,6,1))
    ax[2].set_xlabel('epoch')

    for i in range(3):
        if i ==0:
            color='r'
        elif i==1:
            color='b'
        elif i==2:
            color='g'
        ax[0].plot(index,np.array(acc_list[i]),c=color,label='seed %d'%(i+1))
        ax[1].plot(index,loss[i],c=color,label='seed %d'%(i+1))
        ax[2].plot(index,gradient[i],c=color,label='seed %d'%(i+1))

    ax[0].legend(loc="upper left")
    ax[1].legend(loc="lower left")
    ax[2].legend(loc="lower left")
    fig.tight_layout()
    plt.suptitle(name,x=0.1)
    fig.set_size_inches(5, 7)
    plt.savefig(path)
    #plt.show()

def read_file(name):
    def tf_data(data):
        for i in range(len(data)):
            data[i].replace('\n','')
            index = data[i].find(':')
            data[i] = float(data[i][index+1:])
        return data
    epoch=[]
    train_acc_list=[]
    train_loss_list=[]
    train_gradient_list=[]
    val_acc=[]
    val_f1=[]
    with open(name,'r') as f:
        lines=f.readlines()
    for i in range(len(lines)):
        if lines[i].find('new_epoch')!=-1:
            epoch.append(i)
    for i in range(len(epoch)):
        train_acc=[]
        train_loss=[]
        train_gradient=[]
        if i != len(epoch)-1:
            for j in range(epoch[i],epoch[i+1]):
                if lines[j].find('train acc')!=-1:
                    train_acc.append(lines[j])
                if lines[j].find('train loss')!=-1:
                    train_loss.append(lines[j])
                if lines[j].find('train gradient')!=-1:
                    train_gradient.append(lines[j])
                if lines[j].find('val acc')!=-1:
                    val_acc.append(lines[j])
                if lines[j].find('val f1')!=-1:
                    val_f1.append(lines[j])
        else:
            for j in range(epoch[i],len(lines)):
                if lines[j].find('train acc')!=-1:
                    train_acc.append(lines[j])
                if lines[j].find('train loss')!=-1:
                    train_loss.append(lines[j])
                if lines[j].find('train gradient')!=-1:
                    train_gradient.append(lines[j])
                if lines[j].find('val acc')!=-1:
                    val_acc.append(lines[j])
                if lines[j].find('val f1')!=-1:
                    val_f1.append(lines[j])
        train_acc_list.append(tf_data(train_acc))
        train_loss_list.append(tf_data(train_loss))
        train_gradient_list.append(tf_data(train_gradient))
    val_acc = tf_data(val_acc)
    val_f1 = tf_data(val_f1)
    return val_acc,train_loss_list,train_gradient_list
    #plot(train_acc_list,train_loss_list,train_gradient_list)
    #plot_2(path,name2,val_acc,train_loss_list,train_gradient_list)

def plot_experiments():
    files=os.listdir('./experiments')
    for i in files:
        val_acc,train_loss_list,train_gradient_list = read_file('./experiments/%s/log.txt'%i,)
        plot_2('./experiments/%s/%s.png'%(i,i),i,val_acc,train_loss_list,train_gradient_list)

def plot_experiments_2():
    files=os.listdir('./experiments')
    files.sort()
    for i in range(4):
        val_acc1,train_loss_list1,train_gradient_list1=read_file('./experiments/'+files[0+i]+'/log.txt')
        val_acc2,train_loss_list2,train_gradient_list2=read_file('./experiments/'+files[4+i]+'/log.txt')
        val_acc3,train_loss_list3,train_gradient_list3=read_file('./experiments/'+files[8+i]+'/log.txt')
        va_list,tl_list,tg_list = [val_acc1,val_acc2,val_acc3],[train_loss_list1,train_loss_list2,train_loss_list3],[train_gradient_list1,train_gradient_list2,train_gradient_list3]
        name = ['0_16','0_32','3_16','3_32']
        plot_3('./%d'%i,name[i],va_list,tl_list,tg_list)
if __name__ == '__main__':
    plot_experiments_2()
    #plot_experiments()
    #read_file('/log.txt','./log.png','log')
