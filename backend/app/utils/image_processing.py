import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

async def process_image(file: UploadFile) -> np.ndarray:
    """
    Yüklenen görüntüyü model için uygun formata dönüştürür
    """
    # Dosya içeriğini oku
    contents = await file.read()
    
    # Bytes'ı PIL Image'a dönüştür
    image = Image.open(io.BytesIO(contents))
    
    # Görüntüyü yeniden boyutlandır (model için 224x224)
    image = image.resize((224, 224))
    
    # RGB'ye dönüştür
    image = image.convert('RGB')
    
    # Numpy dizisine dönüştür ve normalize et
    image_array = np.array(image) / 255.0
    
    # Model için boyut ekle (batch boyutu için)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array