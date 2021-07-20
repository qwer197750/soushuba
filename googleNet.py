# library
# standard library
import os
from PIL import Image
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import transforms, datasets
import json
from torchvision.models.googlenet import GoogLeNet
import cv2

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = {
        "train": transforms.Compose([transforms.Resize(28),
                                     transforms.RandomResizedCrop(28),
                                     # transforms.RandomHorizontalFlip(),
                                     # transforms.Grayscale(),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.5], [0.5])
                                     ]),
        "predict": transforms.Compose([transforms.Resize(28),
                                       transforms.CenterCrop(28),
                                       # transforms.Grayscale(),
                                       transforms.ToTensor(),
                                       # transforms.Normalize([0.5], [0.5])
                                       ])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # get data root path
    # image_path = data_root + "flower_data/flower_data/"  # flower data set path
    image_path = "img/"  # flower data set path

    train_dataset = datasets.ImageFolder(root=image_path + "train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=image_path + "predict",
                                            transform=data_transform["predict"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=0)
    pre_weight = torch.load('googlenet.pth')
    model = GoogLeNet(num_classes=24, init_weights=False)
    # model.load_state_dict(pre_weight, strict=False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    step_total = len(train_loader)
    running_loss = 0.0
    Epoch = 100
    best_acc = 0.0
    for epoch in range(Epoch):
        print('epoch: {:}'.format(str(epoch)))
        for step, (image, label) in enumerate(train_loader):
            pred, aux2, aux1 = model(image.to(device))
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()
        acc = 0.0
        with torch.no_grad():
            for vd in validate_loader:
                imgs, labels = vd
                outputs = model(imgs.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs[0], dim=1)[1]
                acc += (predict_y == labels.to(device)).sum().item()
            val_accurate = acc / val_num
        print('acc:{:}'.format(str(val_accurate)))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), 'googlenet.pth')


def predict(imgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(28),
                        transforms.CenterCrop(28),
                        # transforms.Grayscale(),
                        transforms.ToTensor(),
                        # transforms.Normalize([0.5], [0.5])
                        ])
    class_indict = None
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)
    model = GoogLeNet(num_classes=24, init_weights=False)
    # load model weights
    model_weight_path = "googlenet.pth"
    model.load_state_dict(torch.load(model_weight_path))
    model.eval()
    codes = []
    for i in range(len(imgs)):
        img = imgs[i].convert('RGB')
        x = data_transform(img).unsqueeze(0)
        output = model(x.to(device))[0]
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        c = str(class_indict[str(predict_cla)]).upper()
        codes.append(c)
        print("\r预测进度：{:}|{:}".format(i, len(imgs)), end="")
    print()
    return codes


if __name__ == '__main__':
    train()
