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
trn = config.traininglogfilename
val = config.validationlogfilename

trndata = pd.read_csv(trn)
valdata = pd.read_csv(val)
plot_training_validation_metrics(trndata['loss'][10:], valdata['loss'][10:])
# %%
