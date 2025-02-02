import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


rf_model = joblib.load("health_model.pkl")
one_hot_encoder = joblib.load("encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")


app = FastAPI()


class HealthInput(BaseModel):
    skin_tone: str
    wrinkles: str
    dark_circles: str
    spots: str
    face_shape: str
    nail_color: str
    hair_coverage: str
    hairline_status: str

@app.post("/predict")
def predict_health_condition(input_data: HealthInput):
    
    input_df = pd.DataFrame([input_data.dict()])
    
    
    encoded_input = one_hot_encoder.transform(input_df)
    
    
    prediction = rf_model.predict(encoded_input)
    predicted_condition = label_encoder.inverse_transform(prediction)[0]
    
    return {"predicted_health_condition": predicted_condition}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
