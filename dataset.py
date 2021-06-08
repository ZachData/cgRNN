
import os
import time
import string
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from sklearn import preprocessing

# encode letters so that nn can understand it
printable = []
for i in string.printable:
    printable.append(i)
printable.append('blank') #72
le = preprocessing.LabelEncoder()
le.fit(printable)



imsize = 1, 650, 650
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = r'C:\Users\F\Desktop\Snik\Datasets\IAM\formsA-D'
Data_path = r'C:\Users\F\Desktop\Snik\Datasets\IAM\ascii\words.txt'
batch_size = 28


'''
batch does not like variable length inputs, so I must pad using collate_fcn
'''
def collate_fn_padd(batch):
    '''
    Trivial redefinition.
    The actual collation goes in the trainer.
    '''
    ## get sequence lengths
    img     = [batch[i][0] for i in range(len(batch))]
    letters = [batch[i][1] for i in range(len(batch))]
    
    # print(batch_targets)
    return img, letters



class IAM_Dataset(Dataset):
    def __init__(self, img_path, Data_path, transform=None):
        self.img_path = img_path
        self.Data_path = Data_path
        self.transform = transform
        self.images = os.listdir(img_path)
        self.df = pd.read_csv(Data_path, 
                        skiprows=18, 
                        sep=' ', 
                        header=None, 
                        dtype=str, 
                        names=[i for i in range(0, 10)],
                        keep_default_na=False
                        )
        self.df[8] = self.df[8] + self.df[9] # M Ps --> MPs
        self.df = self.df.drop(columns=[9])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # get page, then get letters on the page
        img = os.listdir(img_path)[index]
        idx = 0
        boxes = []
        letters = []
        for name in self.df[0]:
            if name[:-6] == img[:-4]:
                box = (int(self.df[3].iloc[idx]), #[[index], [col]]
                       int(self.df[4].iloc[idx]), 
                       int(self.df[5].iloc[idx]), 
                       int(self.df[6].iloc[idx]))
                boxes.append(box)
                word = self.df[8].iloc[idx]
                for letter in word: # ('MOVE',)
                    #add letter, but encode it first
                    letters.append(le.transform([letter]))
                letters.append(le.transform([' '])) #edit this later. A punctuation mark should remove the last space.
            idx += 1
        # print(img[index], index, 'image path')
        img = Image.open(os.path.join(img_path, img))

        #make cropping box to minimize page size
        boxes = pd.DataFrame(boxes)
        boxes[2], boxes[3] = boxes[0] + boxes[2], boxes[1] + boxes[3] 
        img = img.crop((boxes[0].min(axis=0), 
                       boxes[1].min(axis=0), 
                       boxes[2].max(axis=0), 
                       boxes[3].max(axis=0)))
        newsize = (imsize[1], imsize[2])
        img = img.resize(newsize)
        img = img - np.mean(img)
        img = img / np.std(img)

        tensorize = T.Compose([
            T.ToTensor(),
            # T.Normalize(mean=0, std=1, inplace=False)
                  ])
        img = tensorize(img).type(torch.FloatTensor)
        letters = torch.tensor(letters)

        return img, letters


# def train_fcn(loader, model, optimizer, loss_fcn, scaler):
if __name__ == "__main__":
    my_dataset = IAM_Dataset(img_path=img_path, Data_path=Data_path, transform=None)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_padd)
    # loop = tqdm(train_loader)
    num = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(data[0])
        print(targets[0].squeeze())
        for i in targets:
            print(i.squeeze())
        break
