import csv
from pylab import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
import time
from torchvision.models.mobilenet import mobilenet_v2 as MobileNetV2


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(28),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "predict": transforms.Compose([transforms.Resize(28),
                                   transforms.CenterCrop(28),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    net = MobileNetV2(num_classes=24)
    # load pretrain weights
    model_weight_path = "./mobilenet_v2.pth"
    pre_weights = torch.load(model_weight_path)
    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # # freeze features weights
    # for param in net.features.parameters():
    #     param.requires_grad = False

    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    best_acc = 0.0
    save_path = './mobilenet_v2.pth'
    Epoch = 100
    acc_list = []
    lost_list = []
    for epoch in range(Epoch):
        # train
        net.train()
        running_loss = 0.0
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
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

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))
            lost_list.append([epoch+1, running_loss / step])
            acc_list.append([epoch+1, val_accurate])
            with open('lost.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for r in lost_list:
                    writer.writerow(r)
            with open('acc.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                for r in acc_list:
                    writer.writerow(r)

    print('Finished Training')


def show_loss():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    x_axis_data = []
    y_axis_data = []
    with open("lost.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x_axis_data.append(int(row[0]))
            y_axis_data.append(float(row[1]))

    plt.plot(x_axis_data, y_axis_data, 'ro-', markersize=2, color='#4169E1', alpha=0.8, linewidth=0.8, label='损失函数折线图')

    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('损失函数值')
    plt.savefig('lost.jpg')
    plt.show()


def show_acc():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    x_axis_data = []
    y_axis_data = []
    with open("acc.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            x_axis_data.append(int(row[0]))
            y_axis_data.append(float(row[1]))

    plt.plot(x_axis_data, y_axis_data, 'ro-', markersize=2, color='#4169E1', alpha=0.8, linewidth=0.8, label='准确率折线图')

    plt.legend(loc="upper left")
    plt.xlabel('epoch')
    plt.ylabel('准确率')
    plt.savefig('acc.jpg')
    plt.show()


if __name__ == '__main__':
    train()
    show_acc()
    show_loss()