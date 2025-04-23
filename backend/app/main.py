from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.model.classifier import TrashClassifier
from app.utils.image_processing import process_image

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