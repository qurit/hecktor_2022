#%%
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchsummary import summary
import config
# %%

def get_model():
    # load model
    model = models.resnet18(pretrained=True) 

    # # freeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # changing avgpool and fc layers
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128,1),
        nn.Sigmoid()
        )
    
    return model 

# #%%
# model = get_model().to(config.DEVICE)
# summary(model, (1,3,224,224))
# %%
