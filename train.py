import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data_dir = "dataset"
model_save_path = "honey_detector_model.pt"


# Image Transoforms/Processing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Composing transforms...")


# Loading the data set
dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
print("Loading dataset...")

# Splitting into 80% training and 20% validation data
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
print("Splitting dataset...")

# Creating the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16)
print("Creating data loaders...")

# Load MobilenetV2
model = models.mobilenet_v2(pretrained=True)
print("Loading model...")

# Freeze the parameters
for param in model.parameters():
    param.requires_grad = False
print("Freezing parameters...")

# Replace the final layer with ModelV2Net Output
model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2)

# Define loss and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Defining loss and optimizer...")
# Training loop
nn_epochs = 10 # Increase if results are not good
train_losses, validation_losses = [], []

for epoch in range(nn_epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation Phase
    model.eval()
    validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_validation_loss = validation_loss / len(validation_loader)
    validation_losses.append(avg_validation_loss)

    print(f"Epoch {epoch+1}/{nn_epochs}, Train Loss: {avg_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")


# Saving the trained model
torch.save(model.state_dict(), model_save_path)
print("✅ Model saved to", model_save_path)

# Plot the loss curves
plt.plot(train_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()