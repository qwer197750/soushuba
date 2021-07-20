from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__':
    # img = Image.open('img/coed0.png')
    # img_array = np.asarray(img)
    # print(img_array)
    # print(img_array.shape)
    filenames = os.listdir('img/train/b')[:6]
    filenames.extend(os.listdir('img/train/0')[:6])
    filenames.extend(os.listdir('img/train/2')[:6])
    filenames.extend(os.listdir('img/train/3')[:6])
    filenames.extend(os.listdir('img/train/e')[:6])
    filenames.extend(os.listdir('img/train/j')[:6])
    filenames.extend(os.listdir('img/train/r')[:6])
    filenames.extend(os.listdir('img/train/t')[:6])
    filenames.extend(os.listdir('img/train/v')[:6])
    for i in range(54):
        ss = 'b023ejrtv'
        f = filenames[i]
        j = i // 6
        img = mpimg.imread('img/train/{:}/'.format(ss[j])+f, 0)
        plt.subplot(6, 9, i+1)
        plt.xticks([])  # 不显示x轴
        plt.yticks([])  # 不显示y轴
        plt.imshow(img)

    plt.show()