import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
    setPredictions([]);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const uploadPromises = selectedFiles.map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post('https://atiknet.onrender.com/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        return {
          name: file.name,
          imageUrl: URL.createObjectURL(file),
          ...response.data.prediction,
        };
      });

      const completedResults = await Promise.all(uploadPromises);
      setPredictions(completedResults);
    } catch (error) {
      console.error('Hata:', error);
      alert('Resim yüklenirken bir hata oluştu!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-container">
        <header className="app-header">
          <h1>AtıkNet</h1>
          <p>Yapay Zeka Destekli Atık Sınıflandırma Sistemi</p>
        </header>

        <div className="upload-container">
          <label className="upload-button">
            <input
              type="file"
              multiple
              onChange={handleFileChange}
              accept="image/*"
            />
            <i className="fas fa-cloud-upload-alt"></i>
            <span>Görüntü Seç ({selectedFiles.length} dosya seçildi)</span>
          </label>
          <button 
            onClick={handleSubmit} 
            disabled={!selectedFiles.length || loading}
            className={`predict-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>İşleniyor...</span>
              </>
            ) : (
              'Tahmin Et'
            )}
          </button>
        </div>

        <div className="predictions-grid">
          {predictions.map((pred, index) => (
            <div key={index} className="prediction-card">
              <div className="image-container">
                <img src={pred.imageUrl} alt={`Görüntü ${index + 1}`} />
              </div>
              <div className="prediction-content">
                <h3 className="file-name">{pred.name}</h3>
                <div className="prediction-type">
                  <span className="type-label">Tür:</span>
                  <span className="type-value">{pred.sinif}</span>
                </div>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{width: `${pred.guven * 100}%`}}
                  ></div>
                  <span className="confidence-text">
                    %{(pred.guven * 100).toFixed(2)}
                  </span>
                </div>
                <p className="prediction-message">{pred.mesaj}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
