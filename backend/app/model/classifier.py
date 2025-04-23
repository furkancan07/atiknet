import tensorflow as tf
import numpy as np

class TrashClassifier:
    def __init__(self):
        # Sınıf etiketleri ve Türkçe karşılıkları
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.classes_tr = {
            'cardboard': 'Karton',
            'glass': 'Cam',
            'metal': 'Metal',
            'paper': 'Kağıt',
            'plastic': 'Plastik',
            'trash': 'Diğer Atık'
        }
        
        try:
            # Eğitilmiş modeli yüklemeyi dene
            self.model = tf.keras.models.load_model('trained_model.h5')
            print("Eğitilmiş model başarıyla yüklendi!")
        except:
            print("Eğitilmiş model bulunamadı, yeni model oluşturuluyor...")
            self.model = self._create_model()
        
    def _create_model(self):
        """
        Basit bir CNN modeli oluşturur
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, train_labels, epochs=50, batch_size=32):
        """
        Modeli eğitir
        """
        history = self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Eğitilmiş modeli tam yol ile kaydet
        import os
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trained_model.h5')
        self.model.save(model_path)
        print(f"Model şu konuma kaydedildi: {model_path}")
        return history
    
    def predict(self, image):
        """
        Görüntü sınıflandırması yapar
        """
        predictions = self.model.predict(image)
        predicted_class = self.classes[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return {
            "sinif": self.classes_tr[predicted_class],
            "guven": confidence,
            "mesaj": f"Bu görüntü %{round(confidence * 100, 2)} oranında {self.classes_tr[predicted_class]} sınıfına aittir."
        }