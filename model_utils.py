import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import torch.nn.functional as F

# define convolutional neural network
class CNNNet(torch.nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

# data loading
def get_data_loader(is_train):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_set = datasets.MNIST("", train=is_train, transform=transform, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=is_train)

# evaluation function
def evaluate(test_data, net):
    net.eval()  # switch to evaluation mode
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net(x)
            predicted = torch.argmax(outputs, dim=1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

# save and load model
def save_model(net, path="mnist_model.pth"):
    torch.save(net.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="mnist_model.pth"):
    net = CNNNet()
    net.load_state_dict(torch.load(path))
    net.eval() # switch to evaluation mode
    print(f"Model loaded from {path}")
    return net

# train model
def train_model(epochs=5, path="mnist_model.pth"):
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    net = CNNNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print("Initial accuracy:", evaluate(test_data, net))

    for epoch in range(epochs):
        net.train()  # switch to training mode
        for (x, y) in train_data:
            optimizer.zero_grad() # clear gradients
            output = net(x) # forward pass, output is obtained by passing data through the model
            loss = torch.nn.functional.nll_loss(output, y) # negative log-likelihood loss
            loss.backward() # backward pass, compute gradients from loss
            optimizer.step() # update weights and bias based on the computed gradients
        accuracy = evaluate(test_data, net) # after each epoch, evaluate the model on the test data
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    save_model(net, path)

def process_and_predict(image, model):
    """
    Processes a PIL.Image directly and predicts the digit using the given model.

    Args:
        image (PIL.Image): Input image to be processed and recognized.
        model (torch.nn.Module): Pre-trained PyTorch model for prediction.

    Returns:
        int: Predicted digit.
    """
    try:
        # Step 1: Convert the PIL image to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')  # Ensure grayscale

        # Step 2: Invert colors (MNIST uses white digits on black background)
        image = ImageOps.invert(image)

        # Convert the PIL image to a NumPy array for further processing
        img_np = np.array(image)

        # Step 3: Binarize the image
        _, img_np = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)

        # Step 4: Find the bounding box of the digit (remove unnecessary borders)
        coords = cv2.findNonZero(img_np)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img_cropped = img_np[y:y + h, x:x + w]
        else:
            img_cropped = np.zeros((28, 28), dtype=np.uint8)  # Blank image if no digit is found

        # Step 5: Resize with preserved aspect ratio
        h, w = img_cropped.shape
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))

        img_resized = cv2.resize(img_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center the resized digit on a 20x20 canvas
        canvas_20x20 = np.zeros((20, 20), dtype=np.uint8)
        x_offset = (20 - new_w) // 2
        y_offset = (20 - new_h) // 2
        canvas_20x20[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        # Step 6: Place the 20x20 canvas onto a blank 28x28 canvas
        img_canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        img_canvas[y_offset:y_offset + 20, x_offset:x_offset + 20] = canvas_20x20

        # Save intermediate debugging images
        debug_image_20x20 = Image.fromarray(canvas_20x20)
        debug_image_20x20.save("debug_20x20_canvas.png")
        debug_image_28x28 = Image.fromarray(img_canvas)
        debug_image_28x28.save("uploaded_processed_image.png")

        # Step 7: Normalize and convert to a tensor
        img_tensor = np.expand_dims(img_canvas, axis=0)  # Add channel dimension
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
        img_tensor = torch.from_numpy(img_tensor).float() / 255.0  # Normalize to [0, 1]

        # Step 8: Perform inference with the model
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = model(img_tensor)
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        return pred

    except Exception as e:
        print(f"Error during processing and prediction: {e}")
        return None