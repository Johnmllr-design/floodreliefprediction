from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from fastapi import Request
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

class TurbineObject(BaseModel):
    waterDepth : str
    floodWaterDuration : str
    causeOfDamage : str
    floodEvent : str



@app.post("/prediction")
def Request(req : TurbineObject):
    input_value = [float(req.waterDepth), float(req.floodWaterDuration), float(req.causeOfDamage), dictionary[req.floodEvent][0], dictionary[req.floodEvent][1], dictionary[req.floodEvent][2]]
    input_value = torch.tensor(input_value, torch.float32)
    return {"inference" : model.forward(input_value)}
   