import os

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cuda:1'
experiment_code = 'unet_resnet34enc_diceloss_smalldataoverfit'
mindim_x = 128
mindim_y = 128
SAVE_MODEL_ROOT = '/data/blobfuse/saved_models_hecktor/segmentation'
SAVE_MODEL_DIR = os.path.join(SAVE_MODEL_ROOT,'saved_models_unet_diceloss')
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
traininglogfilename = 'trainlog_'+ experiment_code + '.csv'
validationlogfilename = 'validlog_' + experiment_code + '.csv'