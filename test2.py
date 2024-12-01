import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

# 定义卷积神经网络
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

# 数据加载
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

# 评估函数
def evaluate(test_data, net):
    net.eval()  # 切换到评估模式
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net(x)
            predicted = torch.argmax(outputs, dim=1)
            n_correct += (predicted == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

# 保存和加载模型
def save_model(net, path="mnist_model.pth"):
    torch.save(net.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="mnist_model.pth"):
    net = CNNNet()
    net.load_state_dict(torch.load(path))
    net.eval()
    print(f"Model loaded from {path}")
    return net

# 训练模型
def train_model(epochs=5, path="mnist_model.pth"):
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)

    net = CNNNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print("Initial accuracy:", evaluate(test_data, net))

    for epoch in range(epochs):
        net.train()  # 切换到训练模式
        for (x, y) in train_data:
            optimizer.zero_grad()
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        accuracy = evaluate(test_data, net)
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}")

    save_model(net, path)

# 图像预处理函数
def preprocess_image(image):
    # 转换为灰度图像并反转颜色
    image = image.convert('L')
    image = ImageOps.invert(image)
    # 转换为NumPy数组
    image_np = np.array(image)
    # 二值化处理
    _, image_np = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY)
    # 查找非零像素的坐标
    coords = cv2.findNonZero(image_np)
    # 防止没有找到非零像素
    if coords is not None:
        # 获取包围非零像素的矩形
        x, y, w, h = cv2.boundingRect(coords)
        # 裁剪出数字部分
        image_np = image_np[y:y+h, x:x+w]
    else:
        # 如果未找到非零像素，返回空白图像
        image_np = np.zeros((28, 28), dtype=np.uint8)
    # 调整大小为20x20
    image_np = cv2.resize(image_np, (20, 20), interpolation=cv2.INTER_AREA)
    # 创建28x28的空白图像
    new_image = np.zeros((28, 28), dtype=np.uint8)
    # 将数字粘贴到中心
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    new_image[y_offset:y_offset+20, x_offset:x_offset+20] = image_np
    # 应用腐蚀操作，减小笔画粗细
    kernel = np.ones((2, 2), np.uint8)
    new_image = cv2.erode(new_image, kernel, iterations=1)
    # 转换回PIL图像
    image_processed = Image.fromarray(new_image)
    return image_processed

# 预测图像
def predict_image(image, net):
    # 预处理图像
    image = preprocess_image(image)
    # 保存预处理后的图像以供调试
    image.save("processed_image.png")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(image).unsqueeze(0)  # 添加批次维度，形状为[1, 1, 28, 28]
    with torch.no_grad():
        output = net(img_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# GUI 应用程序
class DigitRecognizerApp:
    def __init__(self, root, model_path="mnist_model.pth"):
        self.root = root
        self.root.title("手写数字识别")

        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(root, text="识别", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=10)

        self.clear_button = tk.Button(root, text="清除", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=10)

        self.model = load_model(model_path)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")

    def paint(self, event):
        x, y = event.x, event.y
        radius = 8  # 根据需要调整笔画粗细
        # 在Canvas上绘制
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black")
        # 在self.image上绘制
        draw = ImageDraw.Draw(self.image)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")

    def predict(self):
        prediction = predict_image(self.image, self.model)
        messagebox.showinfo("预测结果", f"模型预测结果是：{prediction}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")

if __name__ == "__main__":
    mode = input("请选择模式：训练(t) 或 使用(u): ").strip().lower()
    model_path = "mnist_model.pth"

    if mode == "t":
        train_model(epochs=5, path=model_path)
    elif mode == "u":
        root = tk.Tk()
        app = DigitRecognizerApp(root, model_path=model_path)
        root.mainloop()
    else:
        print("无效模式，请选择 't' 进行训练 或 'u' 进入交互模式。")
