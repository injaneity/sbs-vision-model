from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference_sdk import InferenceHTTPClient
from app.settings.settings import settings

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=settings.roboflow_api_key
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

class InferenceResponse(BaseModel):
    predictions: list
    image: str

@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def run_inference():
    """
    Endpoint to upload an image and run inference using InferenceHTTPClient.
    """
    try:

        result = CLIENT.infer("images/7_png.rf.4b1cc77a5fefaae3ec5e662d6d7b492f.jpg", 
                              model_id="mola-vi-in-car/4")
        print(result)

        if result.status_code != 200:
            raise HTTPException(status_code=result.status_code, detail=result.text)

        return {
            "predictions": result.json().get("predictions", []),
            "image": result.json().get("image", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))