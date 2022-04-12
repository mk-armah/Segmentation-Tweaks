import os
import cv2
import numpy as np
import torch
import torchvision
from utils import PARAMS



class SegDataset(torch.utils.data.Dataset):
  def __init__(self,imagePath,maskPath,transform = True,train:bool = True):
    self.imagePath = imagePath
    self.maskPath = maskPath
    self.train = train

    self.images = os.listdir(self.imagePath)
    self.masks = os.listdir(self.maskPath)
    self.transform = transform

  def __len__(self):
    return len(self.images)
  
  
  def transform_data(self,image):
    transform = {"train": torchvision.transforms.Compose([(torchvision.transforms.ToPILImage()),
                                             (torchvision.transforms.Resize((224,224))),
                                                        (torchvision.transforms.ToTensor())
                                                        (torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                          (0.229, 0.224, 0.225)))]),
               
                 #construct a dictionary for preprocessing validation and testing images
                "val/test": torchvision.transforms.Compose([T.Resize(size =(224,224)),
                                                           (T.ToTensor()),
                                                           (T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)))])}
    

    return transform['train' if self.train else "val/test"](image)

  
  def __getitem__(self,idx):
    #read image and mask from directory
    image,mask = os.path.join(self.imagePath,self.images[idx]),os.path.join(self.maskPath,self.images[idx])
    
    image = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB) #read image and convert to RGB using opencv
    mask = cv2.imread(mask,0) #read mask and convert to RGB using opencv

    if self.transforms == True: #if transforms are applied;
      image = transform_data(image,train = self.transform)
      mask = transform-data(mask,train = self.transform)

      return image,mask

    else: #if no transforms are applied
      return image,mask
      
    
def RandomSplitter(dataset):
  idx = list(range(len(dataset)))
  np.random.shuffle(idx)

  #percentage to split for validation = 0.2% of training data
  size_of_val_data = int(len(dataset)*0.2)

  val_data_idx = idx[:size_of_val_data] #validation data starts from the begining of the shuffled index to the size_of_val_data
  train_data_idx = idx[size_of_val_data:]  #training data starts from the size_of_val_data to the end of the suffled list

  val_data_sample = torch.utils.data.SubsetRandomSampler(val_data_idx)
  train_data_sample = torch.utils.data.SubsetRandomSampler(train_data_idx)
  
  return val_data_sample,train_data_sample
    
    

def data_loader(imagePath:str,maskPath:str,train:bool,transform_:bool = True):
  """constructs a dataloader for both training and testing data"""
  dataset = SegDataset(imagePath,maskPath,transform = transform_,train = train)
 
  loader = (torch.utils.data.DataLoader(dataset,PARAMS.batch_size = batch_size,sampler = RandomSplitter(dataset)[0]),
  torch.utils.data.DataLoader(dataset,PARAMS.batch_size = batch_size,sampler = RandomSplitter(dataset)[1])) if train else torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = False)
  
  return loader

if __name__ == "__main__":
  imagePath = PARAMS.train_dir['image']
  maskPath = PARAMS.train_dir['mask']
  dataloader(imagePath =imagePath,maskPath = maskPath,transform_ = True)  
  print("done")