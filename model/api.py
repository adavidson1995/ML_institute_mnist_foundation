import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import MNISTNet

app = FastAPI()


class PredictRequest(BaseModel):
    image: list  # Expecting a 28x28 nested list (grayscale)


# Load model at startup
model = MNISTNet()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


@app.post("/predict")
def predict(req: PredictRequest):
    # Convert input to numpy array and check shape
    arr = np.array(req.image, dtype=np.float32)
    if arr.shape != (28, 28):
        raise HTTPException(
            status_code=400, detail="Input must be 28x28 grayscale image."
        )
    # Normalize and add batch/channel dims
    tensor = transform(arr).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    with torch.no_grad():
        output = model(tensor)
        conf, pred = torch.max(torch.exp(output), 1)
        return {"prediction": int(pred.item()), "confidence": float(conf.item())}
