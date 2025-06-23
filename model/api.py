import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from database import Prediction, get_db
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from model import MNISTNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class PredictRequest(BaseModel):
    image: list  # Expecting a 28x28 nested list (grayscale)
    true_label: int | None = None  # Optional true label


class PredictionOut(BaseModel):
    id: int
    timestamp: datetime
    predicted_digit: int
    true_label: int | None

    class Config:
        from_attributes = True  # for Pydantic v2, replaces orm_mode


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
def predict(req: PredictRequest, db: Session = Depends(get_db)):
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
            predicted_digit = int(pred.item())
            confidence = float(conf.item())

            logger.info(f"Raw output: {output[0].tolist()}")
            logger.info(f"Probabilities: {probs[0].tolist()}")

            # Only log prediction to database if true_label is provided
            if req.true_label is not None:
                prediction = Prediction(
                    predicted_digit=predicted_digit, true_label=req.true_label
                )
                db.add(prediction)
                db.commit()

            return {"prediction": predicted_digit, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[PredictionOut])
def get_history(db: Session = Depends(get_db)):
    # Return the latest 50 predictions
    return db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(50).all()
