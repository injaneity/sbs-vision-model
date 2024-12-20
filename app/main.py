from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, List
from inference_sdk import InferenceHTTPClient
from app.settings.settings import settings
import zipfile
import io
import os
import tempfile
from PIL import Image

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=settings.roboflow_api_key
)

class MultipleInferenceResponse(BaseModel):
    results: Dict[str, List[dict]] 

@app.post("/infer", response_model=MultipleInferenceResponse, tags=["Inference"])
async def run_inference(file: UploadFile = File(...)):
    """
    Endpoint to run inference on images uploaded as a ZIP file.
    """
    try:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are accepted.")
        
        contents = await file.read()

        with zipfile.ZipFile(io.BytesIO(contents)) as the_zip:
            
            image_files = [
                f for f in the_zip.namelist()
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and
                not f.startswith('__MACOSX/') and
                not os.path.basename(f).startswith('._')
            ]

            if not image_files:
                raise HTTPException(status_code=400, detail="No valid image files found in the ZIP.")

            output = {}

            for image_file in image_files:
                try:
                    with the_zip.open(image_file) as image:
                        image_bytes = image.read()

                    try:
                        Image.open(io.BytesIO(image_bytes)).verify()
                    except Exception:
                        output[image_file] = []
                        continue

                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file)[1]) as tmp_file:
                        tmp_file.write(image_bytes)
                        tmp_file_path = tmp_file.name

                    result = CLIENT.infer(tmp_file_path, model_id="mola-vi-in-car/4")
                    os.remove(tmp_file_path)

                    predictions = result.get('predictions', [])
                    if predictions:
                        output[image_file] = predictions
                    else:
                        output[image_file] = []

                except Exception as e:
                    output[image_file] = []

            return {"results": output}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

