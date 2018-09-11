import numpy as np
import os
import cv2
import scipy.io as sio
import struct
from matplotlib import pyplot as plt
import torch

def resize_image_prior2(image, prior, max_shape=510):
    w, h = image.size
    if max(w, h) > max_shape:
        if w > h:
            image = image.resize([max_shape, int(float(max_shape) / w * h)])
            prior = prior.resize([max_shape, int(float(max_shape) / w * h)])
        else:
            image = image.resize([int(float(max_shape) / h * w), max_shape])
            prior = prior.resize([int(float(max_shape) / h * w), max_shape])

    return image, prior

def resize_image_prior(image, prior, input_size=512):

    image = image.resize([input_size, input_size])
    prior = prior.resize([input_size, input_size])

    return image, prior

def resize_image_prior(image, prior, input_size=512):

    image = image.resize([input_size, input_size])
    prior = prior.resize([input_size, input_size])

    return image, prior

def preprocess(image, prior, input_shape=256):
    x = np.array(image, dtype=np.float32)
    x = x[:, :, ::-1]
    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean, dtype=np.float32)
    x = (x - mean)
    w, h, _ = x.shape
    prior_arr = np.array(prior, dtype=np.float32)
    input = np.zeros([input_shape, input_shape, 4], dtype=np.float32)
    input[:w, :h, :3] = x
    input[:w, :h, 3] = prior_arr

    input = input.transpose([2, 0, 1])

    return input[np.newaxis, :3, :, :], input[np.newaxis, 3, :, :]

def load_weights_from_h5(model, h5_path):
    m_dict = model.state_dict()
    parameter = sio.loadmat(h5_path)
    for name, param in m_dict.items():
        # print(name)
        layer_name, suffix = os.path.splitext(name)
        if layer_name == 'conv1_1_r2':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_new_w'])
            print(name + '-------' + layer_name + '_new_w')
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_new_b'], [-1]))
            print(name + '-------' + layer_name + '_new_b')
        elif layer_name == 'fc8':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_saliency_w'])
            print(name + '-------' + layer_name + '_saliency_w')
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_saliency_b'], [-1]))
            print(name + '-------' + layer_name + '_saliency_b')
        elif layer_name == 'fc8_r2':
            m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter['fc8_saliency_r2_new_w'])
            print(name + '-------' + 'fc8_saliency_r2_new_w')
            m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter['fc8_saliency_r2_new_b'], [-1]))
            print(name + '-------' + 'fc8_saliency_r2_new_b')
        else:

            if suffix == '.weight':
                m_dict[layer_name + '.weight'].data = torch.from_numpy(parameter[layer_name + '_w'])
                print(name + '-------' + layer_name + '_w')
            elif suffix == '.bias':
                m_dict[layer_name + '.bias'].data = torch.from_numpy(np.reshape(parameter[layer_name + '_b'], [-1]))
                print(name + '-------' + layer_name + '_b')
            else:
                print (name)
    model.load_state_dict(m_dict)
    return model

if __name__ == '__main__':
    from models import SCNN

    model = SCNN()

    h5_path = 'tcsvt_model_params.mat'
    load_weights_from_h5(model, h5_path)