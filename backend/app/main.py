from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.model.classifier import TrashClassifier
from app.utils.image_processing import process_image
from typing import Dict, List

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model sınıfımızın bir örneğini oluştur
classifier = TrashClassifier()

# Atık türleri hakkında bilgi
WASTE_TYPES = {
    "plastic": {
        "name": "Plastik",
        "description": "Plastik atıklar geri dönüştürülebilir. Temiz ve kuru olarak ayrı bir torbada toplanmalıdır.",
        "recycling_tips": [
            "Plastik şişeleri sıkıştırarak yer kazanın",
            "Kapakları ayrı toplayın",
            "Etiketleri çıkarın"
        ]
    },
    "glass": {
        "name": "Cam",
        "description": "Cam atıklar sonsuz kez geri dönüştürülebilir. Kırık camları dikkatli bir şekilde toplayın.",
        "recycling_tips": [
            "Kırık camları ayrı bir torbada toplayın",
            "Cam şişeleri durulayın",
            "Metal kapakları ayrı toplayın"
        ]
    },
    "paper": {
        "name": "Kağıt",
        "description": "Kağıt atıklar geri dönüştürülebilir. Islak veya yağlı kağıtlar geri dönüştürülemez.",
        "recycling_tips": [
            "Kağıtları düz bir şekilde toplayın",
            "Yağlı veya ıslak kağıtları ayırın",
            "Zarfları ve plastik pencereleri çıkarın"
        ]
    },
    "metal": {
        "name": "Metal",
        "description": "Metal atıklar geri dönüştürülebilir. Teneke kutular ve alüminyum folyo gibi malzemeleri ayrı toplayın.",
        "recycling_tips": [
            "Kutuları sıkıştırın",
            "Temiz ve kuru olarak toplayın",
            "Farklı metal türlerini ayırın"
        ]
    }
}

@app.get("/waste-types")
async def get_waste_types() -> Dict[str, Dict]:
    """
    Tüm atık türleri hakkında bilgi döndürür.
    """
    return WASTE_TYPES

@app.get("/waste-types/{waste_type}")
async def get_waste_type_info(waste_type: str) -> Dict:
    """
    Belirli bir atık türü hakkında detaylı bilgi döndürür.
    """
    if waste_type not in WASTE_TYPES:
        return {"error": "Geçersiz atık türü"}
    return WASTE_TYPES[waste_type]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Görüntüyü işle
        image = await process_image(file)
        
        # Tahmin yap
        prediction = classifier.predict(image)
        
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)