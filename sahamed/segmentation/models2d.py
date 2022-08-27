import torch
import torchvision
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class segmentation_model(nn.Module):
    def __init__(self,model_name,size=128,in_channel=3,num_class=3,out=False):
        super(segmentation_model, self).__init__()  
        if model_name == 'unet_resnet101':
            model = smp.Unet(encoder_name="resnet101",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class,
                             activation='sigmoid')
        if model_name == 'unet_resnet152':
            model = smp.Unet(encoder_name="resnet152",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        if model_name == 'unet_resnet34':
            model = smp.Unet(encoder_name="resnet34",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
            
        if model_name == 'unet_resnet18':
            model = smp.Unet(encoder_name="resnet18",     
                             encoder_weights="imagenet",    
                             in_channels=in_channel,           
                             classes=num_class)
        self.model=model
        self.out=out
    def forward(self, image):
        if self.out:out_mask=self.model(image)['out']
        else: out_mask=self.model(image)
        return out_mask