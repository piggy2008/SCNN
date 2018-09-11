import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from models import SCNN
from utils import preprocess, resize_image_prior2, load_weights_from_h5
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def test(test_dir, test_prior_dir, list_file_path, save_path):
    list_file = open(list_file_path)
    test_names = [line.strip() for line in list_file]

    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    save_path = save_path + '_' + time_str
    device = torch.device('cuda')
    model = SCNN().to(device)

    # test for every model
    # model.load_state_dict(torch.load('model/2018-08-29 09:56:15/10000/snap_model.pth'))
    h5_path = 'tcsvt_model_params.mat'
    load_weights_from_h5(model, h5_path)
    model.eval()
    # model = load_part_of_model(model, 'model/2018-08-22 18:04:08/6000/snap_model.pth')
    src_w = 0
    src_h = 0
    size = 512
    count = 0
    for name in test_names:
        images_path = name.split(',')
        batch_x = np.zeros([4, 3, size, size])
        motion_prior = np.zeros([4, 1, size, size])
        for i, image_name in enumerate(images_path):
            image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
            prior = Image.open(os.path.join(test_prior_dir, image_name + '.png'))

            # src_w, src_h = image.size

            image, prior = resize_image_prior2(image, prior, size)
            src_w, src_h = image.size

            input, prior = preprocess(image, prior, size)

            batch_x[i] = input
            motion_prior[i] = prior

        x = torch.from_numpy(batch_x)
        motion_prior = torch.from_numpy(motion_prior)
        x = x.type(torch.cuda.FloatTensor)
        motion_prior = motion_prior.type(torch.cuda.FloatTensor)
        # feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}
        start = time.clock()
        saliency = model(x, motion_prior)
        saliency = F.sigmoid(saliency)
        end = time.clock()

        count += 1

        saliency = saliency.data.cpu().numpy()
        saliency = saliency * 255
        save_sal = saliency.astype(np.uint8)
        save_img = Image.fromarray(save_sal[3, 0, :src_h, :src_w])
        # save_img = save_img.resize([src_w, src_h])
        # plt.imshow(save_img)
        # plt.show()
        image_path = os.path.join(save_path, images_path[-1] + '.png')
        print('process:', image_path)
        print('time:', end - start)
        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        save_img.save(image_path)


if __name__ == '__main__':
    # test dir
    # FBMS
    # test_dir = '/home/ty/data/FBMS/FBMS_Testset'
    # test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
    # list_file_path = '/home/ty/data/FBMS/FBMS_seq_file.txt'

    # DAVIS
    test_dir = '/home/ty/data/davis/davis_test'
    test_prior_dir = '/home/ty/data/davis/davis_flow_prior'
    list_file_path = '/home/ty/data/davis/davis_test_seq.txt'

    save_path = 'total_result/result_rnn'

    test(test_dir, test_prior_dir, list_file_path, save_path)