import torch

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda:1'
# DATA_DIR = '/data/blobfuse/'
mindim_x = 224
mindim_y = 224
SAVE_MODEL_DIR = '/home/shadab/Projects/bccancer.lymphoma/classification/saved_models_resnet18_focalloss_bccancer_nofreeze/'
