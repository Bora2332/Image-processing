import os     #Dosya ve klasör işlemleri için kullanıyorum
import torch   # Pytorch un ana kütüphanesi
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms   #Görüntü işleme ve veri seti yönetimi için kullanıyorum
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  #Görüntü görselleştirme ve veri manipülasyonu için kullanıyorum
import numpy as np
import torch.nn.functional as F
import random
import shutil   #  Dosya taşıma ve kopyalama işlemleri için kullanıyorum

# Dataset klasör yolunu ayarlayın
dataset_path = r"C:\Users\BORA\Desktop\dataset"

# Kategoriler
class_names = ['Araba', 'Doğa', 'Hayvan', 'Manzara']

# Veri ön işleme (veri augmentasyonu ve normalizasyon)
#RandomHorizontalFlip ve RandomRotation: Görüntüleri rastgele döndürerek çeşitliliği artırır.
#Görseller ImageNet'in ortalama (mean) ve standart sapma (std) değerleriyle normalize edilir normalizasyon yapmak için kullanıyorum.
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet istatistikleri
])

# Eğitim ve doğrulama veri setlerini yüklüyorum
train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'val'), transform=transform)

# Veri yükleyicileri
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Basit CNN Modeli
#3 adet Conv2d: Görüntüden özellik çıkarmak için kullandım
#3 adet MaxPool2d: Boyutları küçültmek için kullandım
#2 adet Linear: Sınıflandırma için tam bağlantılı katmanlar kullandım.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, len(class_names))  # 4 sınıf

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli başlat
#CUDA Kullanımı: GPU varsa cuda cihazını kullanır.
#Kayıp Fonksiyonu: CrossEntropyLoss, çok sınıflı sınıflandırma için.
#Optimizer: Adam, öğrenme oranı 0.001.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modeli eğitme
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Modeli değerlendirme
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

# Modeli kaydetme
torch.save(model.state_dict(), 'simple_cnn.pth')
print("Model kaydedildi: 'simple_cnn.pth'")

# Görselleştirme fonksiyonu
def imshow(img):
    img = img * 0.229 + 0.485  # Normalize geri al
    img = np.clip(img, 0, 1)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()

# Test için rastgele bir örnek seçtirdim
random_index = random.randint(0, len(val_dataset) - 1)
test_img, label = val_dataset[random_index]

# Modeli değerlendirme
model.eval()
with torch.no_grad():
    test_img = test_img.unsqueeze(0).to(device)
    output = model(test_img)

    # Olasılıkları hesapladım
    probabilities = F.softmax(output, dim=1)

    # Tahmin edilen sınıf ve olasılığını hesapladım
    _, pred = torch.max(output, 1)
    predicted_class = class_names[pred.item()]
    predicted_prob = probabilities[0][pred.item()].item() * 100  # Olasılık yüzdesi olarak

    # Gerçek etiket ve tahmin deperlerini gösterdim
    print(f"Gerçek: {class_names[label]}, Tahmin: {predicted_class}, Olasılık: {predicted_prob:.2f}%")

    # Görselleştirme
    imshow(test_img[0])

    # Tüm sınıflar için olasılıkları yazdırdım
    print("Tüm Olasılıklar:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {probabilities[0][i].item() * 100:.2f}%")

# Yeni klasörler oluşturma ve resimleri sınıflara göre yerleştirdim
output_dir = r"C:\Users\BORA\Desktop\sorted_images"  # Yeni klasörün yolu
os.makedirs(output_dir, exist_ok=True)

# Her sınıf için alt klasörler oluşturdum
for class_name in class_names:
    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

# Resimleri sınıflarına göre yerleştirme yaptım
model.eval()
with torch.no_grad():
    for index in range(len(val_dataset)):
        img, label = val_dataset[index]
        img = img.unsqueeze(0).to(device)

        # Modelin tahminini aldım
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]

        # Orijinal resmin yolu ve yeni dosya yolu ayarladım
        img_path = val_dataset.imgs[index][0]
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(output_dir, predicted_class, img_name)

        # Resmi yeni klasöre taşıdım
        shutil.copy(img_path, new_img_path)
        print(f"Resim {img_name} sınıfa '{predicted_class}' yerleştirildi.")
