import os
import zipfile
from flask import Flask, render_template, request, send_file
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class Net(nn.Module):
    '''
        Метод init описывает все слои нейронной сети с параметрами:
            conv1 первый слой на 3 входных канала, 16 выходных. Ядро 3х3
            relu определение функции активации ReLU
            pool определение слоя подвыборки стандартного размера
            conv2 второй слой - 16 входных каналов, 32 выходных. 
            fc1, fc2 линейное преобразование входных данных
        
        Метод forward функция прямого прохода, где x - входные данные

    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # Применение свёрточного слоя, ReLU и слоя подвыборки к данным
        x = self.pool(self.relu(self.conv2(x)))   # Повторное применение
        x = x.view(-1, 32 * 16 * 16)              # Приведение к форме первого полносвязного слоя
        x = self.relu(self.fc1(x))                # Применение ReLU к выходу первого полносвязного слоя
        x = self.fc2(x)                           # Применение второго полносвязного слоя к данным
        return x                                  # Возврат выходных данных сети

app = Flask(__name__)

# Загрузка модели нейросети
model = torch.load('modelThreeClasses.pth')
model.eval()

# Функция для классификации изображения
def classify_image(image_path):
    # Загрузка и предобработка изображения
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)

    # Предсказание класса изображения
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение загруженного файла
        file = request.files['file']
        
        # Создание временной директории для распаковки архива
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        # Сохранение загруженного файла
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)
        
        # Распаковка архива
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Получение списка папок внутри распакованной директории
        subdirectories = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]

        # Обработка изображений и классификация
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(temp_dir, subdirectory)
            
            for root, dirs, files in os.walk(subdirectory_path):
                for file in files:
                    image_path = os.path.join(root, file)
                    class_label = classify_image(image_path)

                    # Проверка, является ли файл изображением, прежде чем его обрабатывать
                    if class_label is not None:
                        # Перемещение изображения в соответствующую папку
                        class_dir = os.path.join(output_dir, str(class_label))
                        os.makedirs(class_dir, exist_ok=True)
                        os.rename(image_path, os.path.join(class_dir, file))
        
        # Создание zip архива с рассортированными изображениями
        output_zip_path = 'output.zip'
        with zipfile.ZipFile(output_zip_path, 'w') as zip_ref:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_ref.write(file_path, os.path.relpath(file_path, output_dir))
    
        # Отправка zip архива пользователю
        return send_file(output_zip_path, as_attachment=True)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)