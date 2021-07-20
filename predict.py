import random

import torch
from torchvision.models.mobilenet import mobilenet_v2 as MobileNetV2
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import json
from cut_image import cut


def predict_code(img_path):
    imgs = cut(img_path)
    code = predict(imgs)
    return ''.join(code).upper()


def predict(imgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(28),
                                   transforms.CenterCrop(28),
                                   transforms.ToTensor()
                                         ])
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = MobileNetV2(num_classes=24)
    # load model weights
    model_weight_path = "mobilenet_v2.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    codes = []
    with torch.no_grad():
        for i in range(len(imgs)):
            img = imgs[i].convert('RGB')
            x = data_transform(img).unsqueeze(0)
            output = model(x.to(device))[0]
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            c = str(class_indict[str(predict_cla)]).upper()
            codes.append(c)
            print("\r预测进度：{:}|{:}".format(i+1, len(imgs)), end="")
        print()
    return codes


if __name__ == '__main__':
    i = random.Random().randint(0, 5000)
    img_path = 'img/source_img/code{:}.png'.format(str(i))
    code = predict_code(img_path)
    print(code)
    img = Image.open(img_path)
    img.show()