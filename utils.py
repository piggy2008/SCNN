import numpy as np
import os
import cv2
import scipy.io as sio
import struct
from matplotlib import pyplot as plt
import torch

def load_weights_from_h5(model, h5_path):
    m_dict = model.state_dict()
    parameter = sio.loadmat(h5_path)
    for name, param in m_dict.items():
        # print(name)
        layer_name, suffix = os.path.splitext(name)

        if suffix == '.weight':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_w'])
            print(name + '-------' + layer_name + '_w')
        elif suffix == '.bias':
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_b'], [-1]))
            print(name + '-------' + layer_name + '_b')
        else:
            print (name)

if __name__ == '__main__':
    from models import SCNN

    model = SCNN()

    h5_path = '/home/ty/tcsvt_model_params.mat'
    load_weights_from_h5(model, h5_path)