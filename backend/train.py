from app.model.classifier import TrashClassifier
from app.model.dataset import DatasetLoader
import os

# Veri seti yolu
data_dir = os.path.join(os.path.dirname(__file__), "dataset")

print("Veri seti yükleniyor...")
dataset = DatasetLoader(data_dir)
X, y = dataset.load_data()

print("\nModel eğitimi başlıyor...")
classifier = TrashClassifier()
history = classifier.train(X, y)

print("\nModel eğitimi tamamlandı!")
print("Eğitilmiş model 'trained_model.h5' olarak kaydedildi.")