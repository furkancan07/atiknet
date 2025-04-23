import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class DatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
    def load_data(self):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            print(f"{class_name} sınıfı yükleniyor...")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                try:
                    # Görüntüyü yükle ve yeniden boyutlandır
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                    img = img.convert('RGB')
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Hata: {img_path} yüklenemedi - {str(e)}")
        
        # Numpy dizilerine dönüştür
        X = np.array(images)
        y = np.array(labels)
        
        # One-hot encoding uygula
        y = to_categorical(y, len(self.classes))
        
        print(f"\nToplam {len(images)} görüntü yüklendi")
        print(f"Veri seti boyutu: {X.shape}")
        
        return X, y