import re
import unicodedata
import string
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
def clean(text):

    #     remove urls
    text = re.sub(r'http\S+', " ", text)

    #     remove mentions
    text = re.sub(r'@\w+',' ',text)

    #     remove hastags
    text = re.sub(r'#\w+', ' ', text)

    #     remove digits
    text = re.sub(r'\d+', ' ', text)

    #     remove html tags
    text = re.sub('r<.*?>',' ', text)
    
    #     remove stop words 
    text = text.strip(' ')
    return text.lower()


all_letters = string.ascii_letters + " .,;'"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

class RNNDATASET(Dataset):
    def __init__(self, array):
        self.data,self.label=array
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index): 
        return (torch.as_tensor(self.data[index]),torch.as_tensor(self.label[index]))
        