import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        ############### R1 ###############
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=(1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=4, dilation=4, padding=(6, 6))
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, dilation=4)
        self.fc8 = nn.Conv2d(4096, 1, kernel_size=1)

        self.pool1_conv = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.pool1_fc = nn.Conv2d(128, 128, kernel_size=1)
        self.pool1_ms_saliency = nn.Conv2d(128, 1, kernel_size=1)

        self.pool2_conv = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))
        self.pool2_fc = nn.Conv2d(128, 128, kernel_size=1)
        self.pool2_ms_saliency = nn.Conv2d(128, 1, kernel_size=1)

        self.pool3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=(1, 1))
        self.pool3_fc = nn.Conv2d(128, 128, kernel_size=1)
        self.pool3_ms_saliency = nn.Conv2d(128, 1, kernel_size=1)

        self.pool4_conv = nn.Conv2d(512, 128, kernel_size=3, padding=(1, 1))
        self.pool4_fc = nn.Conv2d(128, 128, kernel_size=1)
        self.pool4_ms_saliency = nn.Conv2d(128, 1, kernel_size=1)

        ############### R2 ###############
        self.conv1_1_r2 = nn.Conv2d(4, 64, kernel_size=3, padding=(1, 1))
        self.conv1_2_r2 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))

        self.conv2_1_r2 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.conv2_2_r2 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))

        self.conv3_1_r2 = nn.Conv2d(128, 256, kernel_size=3, padding=(1, 1))
        self.conv3_2_r2 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))
        self.conv3_3_r2 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))

        self.conv4_1_r2 = nn.Conv2d(256, 512, kernel_size=3, padding=(1, 1))
        self.conv4_2_r2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv4_3_r2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))

        self.conv5_1_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))
        self.conv5_2_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))
        self.conv5_3_r2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=(2, 2))

        self.fc6_r2 = nn.Conv2d(512, 4096, kernel_size=4, dilation=4, padding=(6, 6))
        self.fc7_r2 = nn.Conv2d(4096, 4096, kernel_size=1, dilation=4)
        self.fc8_r2 = nn.Conv2d(4096, 1, kernel_size=1)

        self.pool1_conv_r2 = nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1))
        self.pool1_fc_r2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pool1_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1)

        self.pool2_conv_r2 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))
        self.pool2_fc_r2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pool2_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1)

        self.pool3_conv_r2 = nn.Conv2d(256, 128, kernel_size=3, padding=(1, 1))
        self.pool3_fc_r2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pool3_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1)

        self.pool4_conv_r2 = nn.Conv2d(512, 128, kernel_size=3, padding=(1, 1))
        self.pool4_fc_r2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pool4_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1)

        self.pool4_conv_r2 = nn.Conv2d(512, 128, kernel_size=3, padding=(1, 1))
        self.pool4_fc_r2 = nn.Conv2d(128, 128, kernel_size=1)
        self.pool4_ms_saliency_r2 = nn.Conv2d(128, 1, kernel_size=1)



    def forward(self, input):

        ############### R1 ###############
        x = F.relu(self.conv1_1(input))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        branch_pool1 = x.clone()

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        branch_pool2 = x.clone()

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.max_pool2d(x, 2)

        branch_pool3 = x.clone()

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.max_pool2d(x, 1)

        branch_pool4 = x.clone()

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.max_pool2d(x, 1)

        x = F.dropout(F.relu(self.fc6(x)), 0.5)
        x = F.dropout(F.relu(self.fc7(x)), 0.5)
        x = self.fc8(x)

        branch_pool1 = F.dropout(F.relu(self.pool1_conv(branch_pool1)), 0.5)
        branch_pool1 = F.dropout(F.relu(self.pool1_fc(branch_pool1)), 0.5)
        branch_pool1 = self.pool1_ms_saliency(branch_pool1)

        branch_pool2 = F.dropout(F.relu(self.pool2_conv(branch_pool2)), 0.5)
        branch_pool2 = F.dropout(F.relu(self.pool2_fc(branch_pool2)), 0.5)
        branch_pool2 = self.pool2_ms_saliency(branch_pool2)

        branch_pool3 = F.dropout(F.relu(self.pool3_conv(branch_pool3)), 0.5)
        branch_pool3 = F.dropout(F.relu(self.pool3_fc(branch_pool3)), 0.5)
        branch_pool3 = self.pool3_ms_saliency(branch_pool3)

        branch_pool4 = F.dropout(F.relu(self.pool4_conv(branch_pool4)), 0.5)
        branch_pool4 = F.dropout(F.relu(self.pool4_fc(branch_pool4)), 0.5)
        branch_pool4 = self.pool4_ms_saliency(branch_pool4)

        up_fc8 = F.upsample_bilinear(x, size=[512, 512])
        up_pool1 = F.upsample_bilinear(branch_pool1, size=[512, 512])
        up_pool2 = F.upsample_bilinear(branch_pool2, size=[512, 512])
        up_pool3 = F.upsample_bilinear(branch_pool3, size=[512, 512])
        up_pool4 = F.upsample_bilinear(branch_pool4, size=[512, 512])

        saliency_predict_r1 = F.sigmoid(up_pool1 + up_pool2 + up_pool3 + up_pool4 + up_fc8)

        ############### R2 ###############
        input_prior = torch.cat((x, saliency_predict_r1), 1)
        x_r2 = F.relu(self.conv1_1_r2(input_prior))
        x_r2 = F.relu(self.conv1_2_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv2_1_r2(x_r2))
        x_r2 = F.relu(self.conv2_2_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv3_1_r2(x_r2))
        x_r2 = F.relu(self.conv3_2_r2(x_r2))
        x_r2 = F.relu(self.conv3_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 2)

        x_r2 = F.relu(self.conv4_1_r2(x_r2))
        x_r2 = F.relu(self.conv4_2_r2(x_r2))
        x_r2 = F.relu(self.conv4_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 1)

        branch_pool4_r2 = x_r2.clone()

        x_r2 = F.relu(self.conv5_1_r2(x_r2))
        x_r2 = F.relu(self.conv5_2_r2(x_r2))
        x_r2 = F.relu(self.conv5_3_r2(x_r2))
        x_r2 = F.max_pool2d(x_r2, 1)

        x_r2 = F.dropout(F.relu(self.fc6_r2(x_r2)), 0.5)
        x_r2 = F.dropout(F.relu(self.fc7_r2(x_r2)), 0.5)
        x_r2 = self.fc8_r2(x_r2)

        branch_pool4_r2 = F.dropout(F.relu(self.pool4_conv_r2(branch_pool4_r2)), 0.5)
        branch_pool4_r2 = F.dropout(F.relu(self.pool4_fc_r2(branch_pool4_r2)), 0.5)
        branch_pool4_r2 = self.pool4_ms_saliency_r2(branch_pool4_r2)

        up_fc8_r2 = F.upsample_bilinear(x_r2, size=[512, 512])
        up_pool4_r2 = F.upsample_bilinear(branch_pool4_r2, size=[512, 512])

        saliency_predict_r2 = up_fc8_r2 + up_pool4_r2

        return saliency_predict_r2

