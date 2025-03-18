from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import uvicorn

app = FastAPI()

# Chargement du modèle et du scaler sauvegardés
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")  # Assure-toi de sauvegarder également le scaler

features = [
    'BB_upper', 'min_price', 'max_price', 'EMA_50', 'open_price'
]

@app.get("/predict")
def predict(
    open_price: float,
    max_price: float,
    min_price: float,
    EMA_50: float,
    BB_upper: float,
    human_time: str
):
    # Créer un DataFrame avec les features
    data = {
        'open_price': [open_price],
        'max_price': [max_price],
        'min_price': [min_price],
        'EMA_50': [EMA_50],
        'BB_upper': [BB_upper],
        'date': [human_time]
    }
    df_features = pd.DataFrame(data, columns=features)
    print(df_features)
    # Mise à l'échelle
    df_scaled = scaler.transform(df_features)
    prediction_date = pd.to_datetime(human_time) + pd.Timedelta(hours=1)
    # Prédiction
    prediction = model.predict(df_scaled)[0]
    return {"predicted_close": prediction,
            "datetime": prediction_date.strftime("%Y-%m-%d %H:%M:%S")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)