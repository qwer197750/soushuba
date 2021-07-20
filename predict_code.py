import requests as requests

from googleNet import predict
from cut_image import cut
from PIL import Image
import csv
import pipreqs.pipreqs

def get_code_img(begin_index, n):
    for i in range(begin_index, begin_index+n):
        a = requests.get('https://meihaoshidai.qiquanshu.com/misc.php?mod=seccode&update=26688&idhash=cS')
        f = open('img\source_img\code{:}.png'.format(i), 'wb')
        f.write(a.content)
        f.close()


def predict_codes(img_paths):
    img_list = []
    for p in img_paths:
        print(p)
        imgs = cut(p)
        img_list.extend(imgs)
    codes = predict(img_list)
    data = []

    j = 0
    for i in range(len(codes)):
        if (i + 1) % 4 == 0 and i != 0:
            code = ''.join(codes[i - 3: i + 1 if i + 1 <= len(codes) else len(codes)])
            data.append(code)
            j += 1
    return data


def predict_to_csv():
    data = []
    csv_file = 'code.csv'
    img_paths = []
    for i in range(0, 5000):
        img_paths.append('img/source_img/code{}.png'.format(str(i)))
    data = predict_codes(img_paths)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        i = 0
        for r in data:
            writer.writerow([i, r])
            i += 1


if __name__ == '__main__':
    # get_code_img(3000, 2000)
    predict_to_csv()