import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as opt
from torch.utils.data.dataloader import DataLoader
from pre import ProcessPdFile
from util import RNNDATASET

class Sentiment(nn.Module):
    def __init__(self,embedding_size,output_size,Vocab_size):
        super(Sentiment,self).__init__()
        self.embedding_dim=embedding_size
        self.output_dim=output_size
        self.Vocab_dim=Vocab_size
        self.emb1=nn.Embedding(self.Vocab_dim,self.embedding_dim)
        self.main=nn.Sequential(nn.Conv1d(self.embedding_dim,256,kernel_size=3,stride=1,padding=0),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=4,stride=2,padding=0),
                                nn.Conv1d(256,128,kernel_size=3,stride=1,padding=0),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=4,stride=2,padding=0),
                                nn.Conv1d(128,64,kernel_size=4,stride=1,padding=0),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=4,stride=2,padding=0)
                                )
        self.fc=nn.Sequential(nn.Linear(64*12,64),
                              nn.ReLU(),
                              nn.Dropout(p=0.1),
                              nn.Linear(64,32),
                              nn.ReLU(),
                              nn.Linear(32,5),
                              nn.Sigmoid()
                              )
    def forward(self ,inputs):
        out1=self.emb1(inputs)
        #out1=out1.unsqueeze(1)
        out2=self.main(out1)
        out3=self.fc(out2.reshape(out2.shape[0],-1))
        return out3
        

def train(train_dl,Net,epochs=30):
    op=opt.Adam(Net.parameters(),lr=0.00005)
    loss_fn=nn.CrossEntropyLoss()
    for i in range(epochs):
        loss_ep=0.0
        acc=0.0
        total=0.0
        for inputs,targets in train_dl:
            op.zero_grad()
            outputs=Net(inputs)
            pred=torch.sum(torch.argmax(outputs,dim=1)==targets)
            loss=loss_fn(outputs,targets)
            loss.backward()
            op.step()
            loss_ep=loss_ep+inputs.shape[0]*loss.data.item()
            acc=acc+pred
            total=total+inputs.shape[0]
        print("Loss for Epoch ",i+1," is ",loss_ep/total," Accuracy for this  Epoch ",acc/total)
        print(total,acc)
        if i%10==0:
            checkpoint={'state_dict':Net.state_dict(),'Epochs':i+1,'opt_dict':op.state_dict()}
            torch.save(checkpoint,'./Rnn.pth')


def test(test_dl,Net):
    acc=0.0
    loss=0.0
    total=0.0
    Net.eval()
    loss_fn=nn.CrossEntropyLoss()
    for inputs,targets in test_dl:
        output=Net(inputs)
        acc=acc+torch.sum(torch.argmax(output,dim=1)==targets)
        print(acc,inputs.shape[0])
        loss=loss+inputs.shape[0]*loss_fn(output,targets)
        total=total+inputs.shape[0]
    
    print("Testing Loss Is ",loss/total," Accuracy of Testing is ",acc/total)





if __name__ == "__main__":
    
    training_path='./archive/Corona_NLP_train.csv'
    testing_path='./archive/Corona_NLP_test.csv'

    pt=ProcessPdFile(training_path)
    pt.MakeDict()
    Vocab_size=pt.data.ReturnSizeofVocab()
    print(Vocab_size)
    array=pt.tokenise()
    train_df=RNNDATASET(array)
    train_dl=DataLoader(train_df,batch_size=32,shuffle=True, num_workers=2)
    print(len(train_df))
    array1=pt.tokenise_test(testing_path)
    test_df=RNNDATASET(array1)
    print(len(test_df))
    test_dl=DataLoader(test_df,batch_size=64,shuffle=True, num_workers=2)

    cp=torch.load('./Rnn.pth')

    
    Net=Sentiment(128,5,Vocab_size)
    Net.load_state_dict(cp['state_dict'])
    
    train(train_dl,Net,epochs=100)
    test(test_dl,Net)