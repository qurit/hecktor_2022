import os 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAIN_DEVICE = 'cuda:0'
# ALL_DEVICE_IDS = [0,1]
experiment_code = 'unet3dmonai_diceloss_fold4'

SAVE_MODEL_ROOT = '/data/blobfuse/saved_models_hecktor/segmentation2'
SAVE_MODEL_DIR = os.path.join(SAVE_MODEL_ROOT,'saved_models_' + experiment_code)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
traininglogfilename = 'trainlog_'+ experiment_code + '.csv'
validationlogfilename = 'validlog_' + experiment_code + '.csv'