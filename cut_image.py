import numpy
import random
import csv
import math
import numpy as np
from PIL import Image
import os


lb = '02346789bcefghjkmprtvwxy'


# 利用裂变的思想，在一个二值矩阵中，找到一个节点数最多的形状（也被称为最大形状）
# 形状指的是矩阵中一些点的集合（这些点都是true，且他们直接是连通的，且矩阵中其它点和这个集合不连通）
# 连通指的是两个点，他们直接存在上下左右、左上右上、左下右下的关系，即称为连通
# 裂变指的是将整个矩阵中所有的true点放到一个待定形状中。这个形状一定包含最大形状
# 然后从待定形状中，找出构成边界的四个点，即含有max_x,max_y,min_x,min_y的四个true点
# 接着从这四个点出发找出这四个点的形状，将这些形状从待定形状中裂解出去
# 重复上述过程，直至裂解后的待定形状小于已知的最大形状，那么已知最大形状为最大形状
class Form():
    def __init__(self):
        # 形状的位置集合
        self.array = []
        self.num = 0
        self.ps = 0
        self.max_x = 0
        self.max_y = 0
        self.min_x = 0
        self.min_y = 0

    # 在一个待定形状中，从指定x，y的点出发，寻找所有的连通点，构成形状
    def init_form_from_ndf(self, form, x, y):
        self.array = form.array
        assert [x, y] in form.ps
        self.ps = [[x, y]]
        form.ps.remove([x, y])
        increase = [-1, 0, 1]
        for p in self.ps:
            [x, y] = p
            for x_ir in increase:
                for y_ir in increase:
                    if x_ir == 0 and y_ir == 0:
                        continue
                    if [x + x_ir, y + y_ir] in form.ps and [x + x_ir, y + y_ir] not in self.ps:
                        self.ps.append([x + x_ir, y + y_ir])
                        form.ps.remove([x + x_ir, y + y_ir])

        self.re_num()
        form.re_num()

    # 在一个矩阵中，找到所有的true点，构成一个待定形状，以便裂解
    def init_no_determined_form(self, array):
        self.array = array
        self.ps = np.transpose(np.nonzero(array)).tolist()
        self.re_num()

    def re_num(self):
        self.num = len(self.ps)
        if self.num <= 0:
            self.max_y = 0
            self.max_x = 0
            self.min_x = 0
            self.min_y = 0
            return
        self.min_x, self.min_y = np.min(np.asarray(self.ps), axis=0)
        self.max_x, self.max_y = np.max(np.asarray(self.ps), axis=0)

    # 裂解待定形状
    def fission(self):
        fs = []
        max_n = 0
        max_form = None
        while max_n < self.num:
            [x, y] = self.ps[0]
            form = Form()
            form.init_form_from_ndf(self, x, y)
            fs.append(form)
            if max_n < form.num:
                max_n = form.num
                max_form = form
        return max_form


def cut(img_path):
    img = Image.open(img_path)
    width, height = img.size
    l = 1/32
    fs = []
    array = np.asarray(img)
    h = 12
    w = 12
    # print(array.shape)
    for i in range(4):
        x_b = i * (1/4) - l
        x_b = x_b if x_b >= 0 else 0
        x_b = int(x_b * width)
        x_e = (i+1) * (1/4) + l
        x_e = x_e if x_e <= 1 else 1
        x_e = int(x_e * width)
        # y_b, y_e = max_position(array[:, x_b:x_e])

        form = Form()
        form.init_no_determined_form(array[:, x_b:x_e])
        max_form = form.fission()
        fs.append([max_form, x_b, x_e])
    i = 0
    res = []
    w, h = array.shape
    for f in fs:
        bk = np.zeros([w+5, h+5], dtype=numpy.bool)
        form, x_b, x_e = f
        for p in form.ps:
            x, y = p
            bk[x+2, x_b+y+2] = array[x, x_b+y]
        res.append(Image.fromarray(bk[:, x_b:x_e+5]))
    return res


def read_label(label_path):
    reader = csv.reader(open(label_path, 'r', encoding='utf8'))
    lables = []
    i = 0
    for r in reader:
        l = r[1]
        if len(l) != 4:
            print(i)
        lables.append(l.strip())
        i = i+1
    return lables


def cutall(path, tran_path, predict_path, label_csv_path, tp=0.2):
    filenames = os.listdir(path)
    string = '02346789bcefghjkmprtvwxy'
    for i in string:
        if not os.path.exists(tran_path+"/"+str(i)):
            os.makedirs(tran_path+"/"+str(i))
        if not os.path.exists(predict_path+'/'+i):
            os.makedirs(predict_path+'/'+i)
    labels = read_label(label_csv_path)

    for i in range(len(filenames)):
        f = filenames[i]
        id = f.replace('code', '')
        id = id.replace('.png', '')
        sub_imgs = cut(path+'/'+f)
        for j in range(len(sub_imgs)):
            img = sub_imgs[j]
            rn = "".join([random.choice(string) for l in range(10)]) + '.png'
            rn = id + rn
            if (i % 100) / 100.0 >= tp:
                img.save('{:}/{:}/{:}'.format(tran_path, labels[int(id)][j], rn))
            else:
                img.save('{:}/{:}/{:}'.format(predict_path, labels[int(id)][j], rn))
        print('\r剪切图片进度{:}|{:}'.format(i+1, len(filenames)), end='')
        if i > 1000:
            return
    print()


if __name__ == '__main__':
    cutall('img/source_img', 'img/train', 'img/predict', 'code.csv')
    # imgs = cut('img/source_img/code1305.png')
    # j = 1
    # for i in imgs:
    #     print(numpy.asarray(i))
    #     i.save('code{:}.png'.format(str(j)))
    #     j += 1