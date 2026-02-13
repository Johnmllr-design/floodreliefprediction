from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from model import flood_comp_model
import uvicorn

app = FastAPI()
model = flood_comp_model()
state = torch.load('model_weights.pt')
dictionary = np.load('event_dictionary.npy', allow_pickle=True).item()
model.load_state_dict(state)
model.eval()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class PredictionRequest(BaseModel):
    waterDepth: float
    floodWaterDuration: float
    causeOfDamage: float
    floodEvent: str

@app.post("/prediction")
def APIcall(req: PredictionRequest):
    # Prepare input for the model
    input_value = [
        float(req.waterDepth),
        float(req.floodWaterDuration),
        float(req.causeOfDamage),
        dictionary[req.floodEvent][0],
        dictionary[req.floodEvent][1],
        dictionary[req.floodEvent][2]
    ]
    print(input_value)
    input_tensor = torch.tensor(input_value, dtype=torch.float32)
    with torch.no_grad():
        inference = model(input_tensor).item()
    return {"inference": inference}





