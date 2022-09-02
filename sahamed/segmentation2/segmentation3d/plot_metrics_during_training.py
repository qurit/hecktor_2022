#%%
import matplotlib.pyplot as plt
import pandas as pd 

import config

def plot_training_validation_metrics(trn_metrics, val_metrics):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.7)

    ax.plot(trn_metrics)
    ax.plot(val_metrics)
    ax.legend(['Training ' ,'Validation'])
    ax.set_xlabel('Epochs')
    ax.set_title('Training and validation losses')
    # fig.savefig('losses_resnet18_focalloss_ptlevelsplit_nocenter_nofreeze.jpg')
    plt.show()


#%%

experiment_code = 'unet3dmonai_diceloss_fold4'
trainlog_file = 'trainlog_'+ experiment_code + '.csv'
validlog_file = 'validlog_'+ experiment_code + '.csv'
# val1 = 'validlog_unet_resnet34enc_diceloss.csv'
# val2 = 'validlog_unet_resnet34enc_diceloss_ct500clip.csv'
trainlogdata = pd.read_csv(trainlog_file)
validlogdata = pd.read_csv(validlog_file)
plot_training_validation_metrics(trainlogdata['loss'], validlogdata['loss'])


   
# %%
