import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.models as models
import multiprocessing
import os
# 使用新的 weights 参数加载预训练的 MobileNetV2 模型
from torchvision.models import MobileNet_V2_Weights
from tqdm import tqdm

BATCH_SIZE = 64

num_classes=2
Learning_rate = 0.01
momentum = 0.5
epochs = 10
if torch.cuda.is_available():
    NUM_WORKERS = 8
else:
    NUM_WORKERS = multiprocessing.cpu_count() - 1
    
    
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=r'比色卡0814\train' ,transform=transform)
test_dataset = datasets.ImageFolder(root=r'比色卡0814\val',transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

# Model setup
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=momentum)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Initialize accuracies list
accuracies = []

# Training function
def train(epoch):
    total = 0.0
    correct = 0.0
    running_loss = 0.0
    model.train()  # Ensure model is in training mode
    for batch_id, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}', unit='batch')):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = torch.argmax(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_id + 1) % 100 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, batch_id + 1, running_loss / 100))
            running_loss = 0.0
            print('Accuracy on train set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    
    accuracy = test()
    accuracies.append(accuracy)
    writer.add_scalar('train accuracy', accuracy, epoch)
    writer.add_scalar('train loss', running_loss, epoch)
    writer.add_graph(model, (inputs,))
    writer.flush()

# Testing function
def test():
    correct = 0
    total = 0
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(test_loader, desc=f'Test', unit='batch')):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    accuracy = 100 * correct / total
    print('Accuracy on test set: %d %% [%d/%d]' % (accuracy, correct, total))
    writer.add_scalar('test accuracy', accuracy, epoch)
    writer.flush()
    return accuracy

# Main execution
if __name__ == '__main__':
    real_epoch = 0
    for epoch in range(epochs):
        real_epoch = epoch + 1
        print(f'Epoch {epoch + 1}/{epochs}')
        print('starting training')
        train(epoch)
        print('starting testing')
        accuracy = test()
        print(f'Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.2f}%')
        if accuracy == 100:
            print('Reached 100% accuracy on test set. Stopping training.')
            break
        

    # Save model
    save_dir = 'models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(save_dir, 'img_recognition_model.pth'))
    
    # Plot accuracy over epochs
    plt.figure()
    plt.plot(range(1, real_epoch+1), accuracies)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    writer.close()
