import pandas as pd
import numpy as np
import util

class Vocab():
    def __init__(self,IsLabel=False):
        self.WordToIndex={}
        self.IndexToWord={}
        self.count=0
        self.TotalWord=0
        self.IsLabel=IsLabel
        if not self.IsLabel:
            self.WordToIndex['<pad>']=1
            self.WordToIndex['<eos>']=2
            self.WordToIndex['<sos>']=0
            self.IndexToWord[1]='<pad>'
            self.IndexToWord[2]='<eos>'
            self.IndexToWord[0]='<sos>'
            self.WordToIndex['<unknown>']=4
            self.IndexToWord[4]='<unknown>'
            self.count=4
            self.TotalWord=4
    def Add_Word(self,word):
        if word not in self.WordToIndex:
            self.WordToIndex[word]=self.count
            self.IndexToWord[self.count]=word
            self.count=self.count+1
        self.TotalWord=self.TotalWord+1
    def MakeVocab(self,listword):
        if self.IsLabel:
            for word in listword:
                self.Add_Word(word)
        else:
            for line in listword:
                for word in line:
                    if word!='':
                        self.Add_Word(word)
    def ReturnSizeofVocab(self):
        return self.count
    def FindTestWordToIndex(self,word):
        if word not in self.WordToIndex:
            return self.WordToIndex['<unknown>']
        else:
            return self.WordToIndex[word]



class ProcessPdFile():
    def __init__(self,path):
        self.Td=pd.read_csv(path)
        self.Label=Vocab(IsLabel=True)
        self.data=Vocab()
    def MakeDict(self):
        t_data=list(self.Td['OriginalTweet'].apply(util.clean).apply(util.unicodeToAscii))
        t_label=list(self.Td['Sentiment'])
        train_data=[]
        for line in t_data:
            line=line.replace('.',' ')
            train_data.append([w for w in line.split(' ') if w!=' 'or w!='.' ])
        self.data.MakeVocab(train_data)
        self.Label.MakeVocab(t_label)
    def tokenise(self):
        train_data=[]
        train_label=[]
        for i in range(self.Td.shape[0]):
            tweet=self.Td['OriginalTweet'].iloc[i]
            sent=self.Td['Sentiment'].iloc[i]
            tweet=util.unicodeToAscii(tweet)
            tweet=util.clean(tweet)
            tweet.replace('.',' ')
            train_t=[word for word in tweet.split(' ') if word!=' ' or word!='.']
            train_d=[w for w in train_t if w!='']
            if(len(train_d)>0):
                train_a=np.ones(128)
                for j in range(len(train_d)):
                    train_a[j]=self.data.FindTestWordToIndex(train_d[j])
                train_data.append(train_a)
                train_label.append(self.Label.WordToIndex[sent])
        return (np.array(train_data,dtype=np.long),np.array(train_label,dtype=np.long))
    def tokenise_test(self,path):
        x=pd.read_csv(path)
        train_data=[]
        train_label=[]
        for i in range(x.shape[0]):
            tweet=x['OriginalTweet'].iloc[i]
            sent=x['Sentiment'].iloc[i]
            tweet=util.unicodeToAscii(tweet)
            tweet=util.clean(tweet)
            tweet.replace('.',' ')
            train_t=[word for word in tweet.split(' ') if word!=' ' or word!='.']
            train_d=[w for w in train_t if w!='']
            if(len(train_d)>0):
                train_a=np.ones(128)
                for j in range(len(train_d)):
                    train_a[j]=self.data.FindTestWordToIndex(train_d[j])
                train_data.append(train_a)
                train_label.append(self.Label.WordToIndex[sent])
        return (np.array(train_data,dtype=np.long),np.array(train_label,dtype=np.long))



