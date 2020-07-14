import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join

green = [0, 128, 0]
blue = [128, 0, 0]
Red = [0, 0, 192]
purple = [128, 0, 64]
pink = [128, 128, 192]
olive = [0, 128, 128]
Wine = [0, 0, 192]

color = []
color.append(green)
color.append(blue)
color.append(Red)
color.append(Wine)
color.append(purple)
color.append(pink)
color.append(olive)

green = [0, 128, 0]
blue = [128, 0, 0]
Red = [0, 0, 192]
purple = [128, 0, 64]
pink = [128, 128, 192]
olive = [0, 128, 128]
Wine = [0, 0, 192]

color = []
color.append(green)
color.append(blue)
color.append(Red)
color.append(Wine)
color.append(purple)
color.append(pink)
color.append(olive)

mypath = '/Users/kwon/Downloads/VOCdevkit/VOC2012/SegmentationObject/'
jpgpath = '/Users/kwon/Downloads/VOCdevkit/VOC2012/JPEGImages/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

try:
    onlyfiles.remove('.DS_Store')
except:
    pass

i = 0
save_index = 0

for f in onlyfiles:
    imgf = f[:-3] + 'jpg'
    filepath = jpgpath + imgf
    segpath = mypath + f
    img = cv2.imread(filepath, 1)
    seg = cv2.imread(segpath, 1)

    h, w, _ = img.shape

    for col in color:
        save_img = np.zeros([h, w, 3])
        f1 = seg[:, :, 0] == col[0]
        f2 = seg[:, :, 1] == col[1]
        f3 = seg[:, :, 2] == col[2]

        f = np.multiply(f1, f2)
        f = np.multiply(f, f3)
        f = np.tile(f, [3, 1, 1])
        f = np.transpose(f, [1, 2, 0])
        save_img = np.multiply(img, f)
        temp = np.max(f)

        if temp == 1:
            find = f[:, :, 0] + f[:, :, 1] + f[:, :, 2]
            width_finding = np.sum(find, axis=0)
            height_finding = np.sum(find, axis=1)

            for i in range(len(width_finding)):
                if width_finding[i] != 0:
                    x_lo = i
                    break

            for i in range(len(width_finding) - 1, -1, -1):
                if width_finding[i] != 0:
                    x_hi = i
                    break

            for i in range(len(height_finding)):
                if height_finding[i] != 0:
                    y_lo = i
                    break

            for i in range(len(height_finding) - 1, -1, -1):
                if height_finding[i] != 0:
                    y_hi = i
                    break

            find = np.tile(find, [3, 1, 1])
            find = np.transpose(find, [1, 2, 0])

            truth_seg = find[y_lo:y_hi + 1, x_lo:x_hi + 1]
            truth_image = save_img[y_lo:y_hi + 1, x_lo:x_hi + 1]

            cv2.imwrite('/Users/kwon/Downloads/croped_seg/%05d.jpg' % save_index, np.float32(truth_seg)*255)
            cv2.imwrite('/Users/kwon/Downloads/croped_save/%05d.jpg' % save_index, truth_image)
            save_index += 1