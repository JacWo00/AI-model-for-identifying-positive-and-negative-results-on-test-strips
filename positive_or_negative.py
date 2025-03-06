import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import os

num_classes = 2
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model_dir = 'models'
model.load_state_dict(torch.load(os.path.join(model_dir, 'positive_or_negative_model.pth')))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def classify_image(image_path):
    # Load and preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
        classes=['阳性','阴性']
        print(f'Predicted class: {classes[predicted_class]}')
        return predicted_class
    
while True:
    print("enter n to exit")
    img_path=input("enter the image path:")
    if img_path=='n':    
        break
    classify_image(img_path)
    