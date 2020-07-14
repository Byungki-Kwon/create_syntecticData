import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from make_background import *


def syntectic_data(backgroundDataset, object_filename, object_path):

    # 물체의 개수
    num_object = np.random.randint(7, 15)
    scale = np.random.uniform(1, 2, num_object+1)  # scale[0]는 배경

    # 임의의 한개의 background index 를 가져옴
    back_index = np.random.randint(0, len(backgroundDataset)-1)
    # object 개수에 맞춰 랜덤한 index 를 가져옴
    object_index = np.random.randint(0, len(object_filename)-1, num_object)

    # mscoco background Image를 가져옴
    back_imag = backgroundDataset.__getitem__(back_index)[0]

    # voc croped 한것들을 가져옴
    temp_object = []

    for f in object_index:
        object_root = object_path + object_filename[f]
        temp = Image.open(object_root)
        temp = np.array(temp, dtype=np.float32)
        temp = temp/255
        temp_object.append(temp)



    c, h, w = back_imag.shape

    # 이미지를 정사각형으로 만들어줌
    length = np.min([h, w])
    print(length)
    croped_back = back_imag[:, 0:length, 0:length]
    croped_back = np.array(croped_back, dtype=np.float32)
    croped_back = np.transpose(croped_back, [1, 2, 0])

    # background에 random scale 적용
    length = int(np.floor(scale[0] * length))

    object_image = []


    for i in range(len(temp_object)):
        h, w, c = temp_object[i].shape
        if np.max([h, w]) < length - 20:
            pad_object = np.zeros([length+20, length+20, 4], dtype=np.float32)
            pad_object[20:20 + h, 20:20 + w, :] = temp_object[i][0:h, 0:w, :]
            object_image.append(pad_object)


    # final image에서의 displace와 이동하는 direction mag_factor 계산
    displace = np.random.uniform(0, 10, num_object + 1) # 물체 + background
    random_theta = np.random.uniform(0, 2*np.pi, num_object + 1)  # 물체 + background

    # displacement를 현재 image를 고려하여 보정
    displace = displace * length/384

    # 진동하는 주기를 랜덤하게 선정해줌 (7~30), +1 을 하는 이유는 처음 프레임과 마지막 프레임을 알아서 계산시키기 위해서임
    frames = np.random.uniform(7, 30, num_object + 1) + 1

    # object의 initial position을 정해줌
    position_x = np.random.randint(0, length, num_object)
    position_y = np.random.randint(0, length, num_object)

    # frame 진동을 위해 설정해둔 인덱스
    index = np.zeros([num_object + 1])

    # warping vector
    warp_vector = np.zeros([num_object+1, 2], dtype=np.float32)

    # base_img 처음에 비어있는 이미지
    base_img = np.zeros([length, length, 3], dtype=np.float32)

    # 몇프레임 뽑아낼 것인가?
    for i in range(60):

        # frame을 오르락 내리락 뽑아내기 위
        for idx in range(len(index)):
            index[idx] = index[idx] + (-1)**(i//frames[idx])
            warp_vector[idx, 1] = - index[idx] * displace[idx] * np.cos(random_theta[idx]) / frames[idx]
            warp_vector[idx, 2] = - index[idx] * displace[idx] * np.sin(random_theta[idx]) / frames[idx]


        for i in range()
        background = backward_warping(base_img, croped_back, warp_vector, scale)


    #     for j in range(1, num_object+1):
    #         foreground, segmentation = make_foreground(object_image[j-1], object_seg[j-1], displace[j],
    #                                                    random_theta[j], frames[j], index[j], scale[j],
    #                                                    position_x[j], position_y[j])
    #
    #         finding = np.sum(segmentation, axis=2)
    #         find_zero = (finding == 0)
    #         ones = np.ones([find_zero.shape[0], find_zero.shape[1]])
    #         find_one = ones - find_zero
    #
    #         background[:, :, 0] = np.multiply(background[:, :, 0], find_zero)
    #         background[:, :, 1] = np.multiply(background[:, :, 1], find_zero)
    #         background[:, :, 2] = np.multiply(background[:, :, 2], find_zero)
    #
    #         foreground[:, :, 0] = np.multiply(foreground[:, :, 0], find_one)
    #         foreground[:, :, 1] = np.multiply(foreground[:, :, 1], find_one)
    #         foreground[:, :, 2] = np.multiply(foreground[:, :, 2], find_one)
    #
    #         background = background + foreground
    #
    #     plt.imshow(background)
    #     plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #


    # row = torch.linspace(-1, 1, w)
    # col = torch.linspace(-1, 1, h)
    # grid = torch.zeros(bn, h, w, 2)
    #
    # for n in range(bn):
    #     for i in range(h):
    #         grid[n, i, :, 0] = row
    #     for i in range(w):
    #         grid[n, :, i, 1] = col
    #
    # grid_image = F.grid_sample(back_imag, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    #
    # grid_image = grid_image.permute(2, 3, 1, 0)
    # grid_image = grid_image.numpy()
    #
    # grid_image = np.squeeze(grid_image)
    # plt.imshow(grid_image)
    # plt.show()