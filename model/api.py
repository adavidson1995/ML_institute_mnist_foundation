import logging
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import MNISTNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class PredictRequest(BaseModel):
    image: list  # Expecting a 28x28 nested list (grayscale)


# Load model at startup
try:
    logger.info("Loading model...")
    model = MNISTNet()
    model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.pth")
    logger.info(f"Loading model from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Convert input to numpy array and check shape
        arr = np.array(req.image, dtype=np.float32)
        logger.info(f"Input shape: {arr.shape}")
        logger.info(f"Input range: [{arr.min()}, {arr.max()}]")

        if arr.shape != (28, 28):
            raise HTTPException(
                status_code=400, detail="Input must be 28x28 grayscale image."
            )

        # Normalize and add batch/channel dims
        tensor = transform(arr).unsqueeze(0)  # Shape: [1, 1, 28, 28]
        logger.info(f"Tensor shape: {tensor.shape}")
        logger.info(f"Tensor range: [{tensor.min()}, {tensor.max()}]")

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            logger.info(f"Raw output: {output[0].tolist()}")
            logger.info(f"Probabilities: {probs[0].tolist()}")
            return {"prediction": int(pred.item()), "confidence": float(conf.item())}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
