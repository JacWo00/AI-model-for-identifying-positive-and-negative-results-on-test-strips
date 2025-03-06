import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import os

num_classes = 2

# 这里没有直接加载模型，而是先加载权重。
# 因为我直接保存和加载模型时，不知为何正确率会降低，目前还没研究出什么原因，所以先用加载权重的方式。

# 如果用cpu的话，此变量改为False
useGPU = True
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model_path = r'D:\onko\图像识别\models\tp_existence_detection_model_dict.v2.11.pth'
model.load_state_dict(torch.load(model_path))

if useGPU:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
    
model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# 预处理图片
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# 进行检测
def detect_existence(image_path):
    # Load and preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        classes=['有试纸','无试纸']
        print(f'Predicted class: {classes[predicted_class]}')
        return classes[predicted_class]


wrongImg1=[]
cnt=0
for imgPath in os.listdir(r'D:\onko\定位裁切图片\processedImg\exist'):
    if detect_existence(r'D:\onko\定位裁切图片\processedImg\exist\\'+imgPath) == '无试纸':
        cnt+=1
        print(imgPath)
        wrongImg1.append(imgPath)
print(f'共检测到{cnt}张无试纸图片')

wrongImg2=[]
cnt=0
for imgPath in os.listdir(r'D:\onko\定位裁切图片\processedImg\notExist'):
    if detect_existence(r'D:\onko\定位裁切图片\processedImg\notExist\\'+imgPath) == '有试纸':
        cnt+=1
        print(imgPath)
        wrongImg2.append(imgPath)
print(f'共检测到{cnt}张有试纸图片')
print(wrongImg1)
print(wrongImg2)

while True:
    print("enter n to exit")
    img_path=input("enter the image path:")
    if img_path=='n':    
        break
    detect_existence(img_path)