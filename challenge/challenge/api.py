import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError
from pydantic import BaseModel

from challenge.model import DelayModel

# Crear una instancia de la clase DelayModel con el modelo entrenado
model = DelayModel()

app = fastapi.FastAPI()

# Definir un modelo Pydantic para las solicitudes de predicción
from pydantic import BaseModel, constr, conint

class PredictionRequest(BaseModel):
    OPERA: constr(min_length=1)
    TIPOVUELO: constr(min_length=1)
    MES: conint(ge=1, le=12)
    

# Endpoint de salud
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# Endpoint para hacer predicciones
@app.post("/predict", status_code=200)
async def post_predict(request_data: PredictionRequest) -> dict:
    # Los datos de entrada ya se han validado automáticamente gracias a FastAPI.
    # request_data contiene los datos validados según el modelo PredictionRequest.

    # Puedes acceder a los campos validados como atributos de request_data:
    OPERA = request_data.OPERA
    TIPOVUELO = request_data.TIPOVUELO
    MES = request_data.MES

    # Realiza la predicción utilizando el modelo DelayModel (debe estar disponible aquí)
    try:
        # Realiza tu predicción utilizando los datos validados
        predictions = model.predict(OPERA, TIPOVUELO, MES)
        # `model` debe ser una instancia de la clase DelayModel con el modelo entrenado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

    # Devuelve el resultado de la predicción
    return {
        "predictions": predictions[0]  # Ajusta esto según cómo devuelva tu modelo las predicciones
    }
