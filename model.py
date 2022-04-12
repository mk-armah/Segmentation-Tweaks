import torchvision
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2


class DownSample(torch.nn.Module):
  def __init__ (self,in_channels,out_channels):
    super(DownSample,self).__init__()

    self.conv = torch.nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride = 1,padding = 1,bias = False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace = True),

                                    nn.Conv2d(in_channels = out_channels,out_channels = out_channels,kernel_size = 3,stride = 1,padding = 1,bias = False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace =True)
    )

  def forward(self,image):
    return self.conv(image)





class DownSample_mn(torch.nn.Module):
  """An Alternative Model For Optimization"""
  def __init__(self,in_channels,out_channels):
    super(DownSample_mn,self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conv = torch.nn.Sequential(torch.nn.Conv2d(in_channels = self.in_channels,out_channels = self.in_channels,
                                                  kernel_size = (3,3),stride = 1,padding = 1,groups = self.in_channels),
                                  torch.nn.ReLU(inplace = True),
                                  torch.nn.Conv2d(in_channels = self.in_channels,out_channels = self.out_channels,
                                                  kernel_size = (1,1)),
                                    torch.nn.BatchNorm2d(self.out_channels),
                                    torch.nn.ReLU(inplace = True),
                                    
                                    torch.nn.Conv2d(in_channels = self.out_channels,out_channels = self.out_channels,
                                                    kernel_size = (3,3),padding = 1,bias = False),
                                    torch.nn.BatchNorm2d(self.out_channels))
    
  def forward(self,image):
    return self.conv(image)


class Unet_101(torch.nn.Module):

  def __init__(self,in_channels = 3,out_channels = 1,features = [64,128,256,512]):
    
    super(Unet_101,self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels


    self.downs = torch.nn.ModuleList() #a list to store all our contraction layers
    self.ups = torch.nn.ModuleList() #a list to store all our expansion layers
    self.maxpool = torch.nn.MaxPool2d(kernel_size = 2,stride = 2) #an intermediate maxpool after each downsample ... to be applied later in the forward
    
    """construct the contraction layers using a for loop and the DownSample class"""

    for feat in features: #for every channel (number) in the features list defined in the init
      down_sample = DownSample(in_channels = self.in_channels,out_channels = feat) #Apply a DownSample
      self.downs.append(down_sample) #append it to the downs list 
      self.in_channels = feat #update the input_channels for the next channel -- note : the output of one downsample layer will be an input to the next

    """construct the expansion layers using a for loop"""

    for feat in reversed(features): # we are reversing the the features list because upsampling starts from 512 to 64
      up_sample = torch.nn.ConvTranspose2d(feat*2,feat,kernel_size = 3,stride = 2)    
      self.ups.append(up_sample)
      self.ups.append(DownSample(feat*2,feat))

    self.BottleNeck =  DownSample(features[-1],features[-1]*2)

    self.lastconv = torch.nn.Conv2d(in_channels = features[0],out_channels = self.out_channels, kernel_size = 1,stride=1)


  def forward(self,image):
    skip_connections = [] #initialize an empty list to store skipped connections
   
    for down in self.downs:
      image = down(image)
      skip_connections.append(image)
      image = self.maxpool(image)

    image = self.BottleNeck(image)
      
      ############################ contracting path is complete ############################

    skip_connections = skip_connections[::-1] #reversing the skipped connections so we can start from the highest number (512) or  the down part of the chart
      
      #concatenate the upsampling and the skipped connections
    for idx in range(0,len(self.ups),2):
        self.ups[idx]
        image = self.ups[idx](image)
        connection = skip_connections[idx//2] #select skip connection index counting from 0 .... len(self.ups) linearly
          
        if image.shape!=connection.shape:
          image = TF.resize(image,size = connection.shape[2:])

        concat_connection = torch.cat((connection,image),dim = 1) #concatenating along the channel dimension ; hence dim = 1.Note ===> we have batch,channel,height,width in torch.cat
        image = self.ups[idx+1](concat_connection)

    return self.lastconv(image)


class Unet_102(torch.nn.Module):

  def __init__(self,in_channels = 3,out_channels = 1,features = [64,128,256,512]):
    
    super(Unet_102,self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels


    self.downs = torch.nn.ModuleList() #a list to store all our contraction layers
    self.ups = torch.nn.ModuleList() #a list to store all our expansion layers
    self.maxpool = torch.nn.MaxPool2d(kernel_size = 2,stride = 2) #an intermediate maxpool after each downsample ... to be applied later in the forward
    
    """construct the contraction layers using a for loop and the DownSample class"""

    for feat in features: #for every channel (number) in the features list defined in the init
      down_sample = DownSample_mn(in_channels = self.in_channels,out_channels = feat) #Apply a DownSample
      self.downs.append(down_sample) #append it to the downs list 
      self.in_channels = feat #update the input_channels for the next channel -- note : the output of one downsample layer will be an input to the next

    """construct the expansion layers using a for loop"""

    for feat in reversed(features): # we are reversing the the features list because upsampling starts from 512 to 64
      up_sample = torch.nn.ConvTranspose2d(feat*2,feat,kernel_size = 3,stride = 2)    
      self.ups.append(up_sample)
      self.ups.append(DownSample_mn(feat*2,feat))

    self.BottleNeck =  DownSample_mn(features[-1],features[-1]*2)

    self.lastconv = torch.nn.Conv2d(in_channels = features[0],out_channels = self.out_channels, kernel_size = 1,stride=1)


  def forward(self,image):
    skip_connections = [] #initialize an empty list to store skipped connections
   
    for down in self.downs:
      image = down(image)
      skip_connections.append(image)
      image = self.maxpool(image)

    image = self.BottleNeck(image)
      
      ############################ contracting path is complete ############################

    skip_connections = skip_connections[::-1] #reversing the skipped connections so we can start from the highest number (512) or  the down part of the chart
      
      #concatenate the upsampling and the skipped connections
    for idx in range(0,len(self.ups),2):
        self.ups[idx]
        image = self.ups[idx](image)
        connection = skip_connections[idx//2] #select skip connection index counting from 0 .... len(self.ups) linearly
          
        if image.shape!=connection.shape:
          image = TF.resize(image,size = connection.shape[2:])

        concat_connection = torch.cat((connection,image),dim = 1) #concatenating along the channel dimension ; hence dim = 1.Note ===> we have batch,channel,height,width in torch.cat
        image = self.ups[idx+1](concat_connection)

    return self.lastconv(image)