import os
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from Pytorch_Image_Classification_main.TransferLearning.utils import plot_history, train
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
    
def main():
    # 定义图像预处理步骤
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Batch_Size = 64

    trainset = datasets.ImageFolder(root=r'比色卡0814\train' ,transform=transform)
    testset = datasets.ImageFolder(root=r'比色卡0814\test',transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size,shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=Batch_Size,shuffle=True, num_workers=8)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MobileNetV2(num_classes=10).to(device)
    
    if device == 'cuda':
        net.to(device)
        net = nn.DataParallel(net)
    
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94 ,patience = 1,min_lr = 0.000001) # 动态更新学习率

    epoch = 10

    if not os.path.exists('./model'):
        os.makedirs('./model')
    else:
        print('文件已存在')
        
    save_path = './model/MoblieNetv2.pth'

    os.makedirs("./logs", exist_ok=True)
    tbwriter = SummaryWriter(log_dir='./logs/MoblieNetv2', comment='MoblieNetV2')  # 使用tensorboard记录中间输出
    tbwriter.add_graph(model=net, input_to_model=torch.randn(size=(64, 3, 224, 224)))

    Acc, Loss, Lr = train(net, trainloader, testloader, optimizer,criterion,scheduler,epoch,save_path, tbwriter, verbose = True)

    plot_history(epoch ,Acc, Loss, Lr)

def imshow(img):
    img = img / 2 + 0.5
    img = np.transpose(img.numpy(),(1,2,0))
    plt.imshow(img)
    plt.show()    

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(224 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
def predict(img,classes,model,device):
        trans = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                    std=(0.5, 0.5, 0.5)),
                            ])
        img = trans(img)
        img = img.to(device)
        # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        img = img.unsqueeze(0)  # 扩展后，为[1，3，32，32]
        output = model(img)
        prob = F.softmax(output,dim=1) #prob是10个分类的概率
        print("概率",prob)
        value, predicted = torch.max(output.data, 1)
        print("类别",predicted.item())
        print(value)
        
        pred_class = classes[predicted.item()]
        print("分类",pred_class)

def predict_CIFAR10():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileNetV2(num_classes=10)

    save_path=r'model\MoblieNetv2.pth'
    model = torch.load(save_path, map_location="cpu")  # 加载模型
    model.to(device)
    model.eval()  # 把模型转为test模式

    # 读取要预测的图片
    img = Image.open("./airplane.png").convert('RGB') # 读取图像
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']
    
    predict(img,classes,model,device)
    
if __name__ == '__main__':
    main()