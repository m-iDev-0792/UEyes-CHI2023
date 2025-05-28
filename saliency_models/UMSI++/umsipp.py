import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import keras
from keras.layers import Input, TimeDistributed, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.path.append('./src')
tf.compat.v1.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)

from util import get_model_by_name

from sal_imp_utilities import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(f'umsipp.py: get_available_devices(): {get_available_devices()}')

print(f'umsipp.py: tf.test.is_gpu_available(): {tf.test.is_gpu_available()}')


def download_weights(W = "./weights/umsi++.hdf5"):
    print(f'umsipp.py: apply_umsipp(): make dir {os.path.dirname(W)}')
    os.makedirs(os.path.dirname(W), exist_ok=True)
    dl_url = "https://userinterfaces.aalto.fi/ueyeschi23/model_weights.zip"
    import requests
    import zipfile
    print(f'Downloading weights from {dl_url}')
    response = requests.get(dl_url)
    if response.status_code != 200:
        raise Exception(f'Failed to download weights from {dl_url}. Status code: {response.status_code}')

    dl_file = "model_weights.zip"
    print(f'download_weights(): download file {dl_file} done')
    with zipfile.ZipFile(dl_file, 'r') as zip_ref:
        zip_ref.extractall("./model_weights")
        
    shutil.copy("./model_weights/saliency_models/UMSI++/umsi++.hdf5", W)
    print(f'download_weights(): copy file {W} done')

def apply_umsipp(file_path, out_path):
    # Init the model
    ckpt_savedir = "ckpt"
    weightspath = ""
    batch_size = 4
    init_lr = 0.0001
    lr_reduce_by = .1
    reduce_at_epoch = 3
    n_epochs = 50
    opt = Adam(lr=init_lr) 
    # losses is a dictionary mapping loss names to weights 
    losses = {
        'kl': 10,
        'cc': -3,
    }

    model_name = "UMSI"
    model_inp_size = (256, 256)
    model_out_size = (512, 512)

    model_params = {
        'input_shape': model_inp_size + (3,),
        'n_outs': len(losses),
    }
    model_func, mode = get_model_by_name(model_name)
    assert mode == "simple"
    model = model_func(**model_params)


    W = "./weights/umsi++.hdf5"
    if not os.path.exists(W):
        print(f'umsipp.py: apply_umsipp(): {W} does not exist, now download it')
        download_weights(W)
    model.load_weights(W)


    img = preprocess_images([file_path], model_inp_size[0], model_inp_size[1])[0]
    basemap = cv2.imread(file_path)
    mapshape = basemap.shape
    print(f'umsipp.py: apply_umsipp(): model.input_shape: {model.input_shape}')
    preds = model.predict(np.expand_dims(img, axis=0))
    preds_map = preds[0]
    preds_classif = preds[1]
    pred_result = postprocess_predictions(np.squeeze(preds_map[0]), mapshape[0], mapshape[1], normalize=False, zero_to_255=True)
    # Normalize to 0-1 range
    pred_result = (pred_result - np.min(pred_result)) / (np.max(pred_result) - np.min(pred_result))
    
    # Apply colormap to get heatmap visualization
    remapped_pred_result = cv2.applyColorMap((pred_result * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # Blend with original image
    alpha = 0.65    
    blended_image = alpha * remapped_pred_result + (1 - alpha) * basemap
    cv2.imwrite(out_path, blended_image)


if __name__ == "__main__":
    apply_umsipp("/Users/hezhenbang/Downloads/Rico-dataset/combined/1.jpg", "/Users/hezhenbang/Desktop/UBCO/Research/PyramidUI/VisualSaliency/VS_Benchmark_Results/1__UMSIpp.jpg")  
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    apply_umsipp(args.file_path, args.out_path)     


