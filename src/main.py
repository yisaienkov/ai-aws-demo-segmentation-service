import io
import logging
from io import BytesIO
from typing import Dict, Union

from fastapi import Response
from PIL import Image
import torch
import numpy as np
from fastapi import FastAPI, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id


class Message(BaseModel):
    message: str


app = FastAPI()
logger = logging.getLogger(__name__)


DEFAULT_RESPONSES: Dict = {
    500: {"description": "Internal Server Error", "model": Message},
}


COLORS = []
for x in range(0, 256, 64):
    for y in range(0, 256, 64):
        for z in range(0, 256, 64):
            COLORS.append([x, y, z])


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")


@app.get("/api/v1/health", response_model=Message, responses={**DEFAULT_RESPONSES})
async def health() -> Message:
    """Health endpoint."""
    return Message(message="Success.")


@app.post("/api/v1/predict", responses={**DEFAULT_RESPONSES})
async def predict(file: bytes = File()) -> Union[Response, JSONResponse]:
    """Predict endpoint."""
    try:
        image = Image.open(io.BytesIO(file))
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg_id = rgb_to_id(panoptic_seg)

        mask = np.zeros_like(panoptic_seg)
        for x in range(panoptic_seg_id.shape[0]):
            for y in range(panoptic_seg_id.shape[1]):
                mask[x, y] = COLORS[panoptic_seg_id[x, y]]

        image = Image.fromarray(mask)

        with BytesIO() as buf:
            image.save(buf, format='PNG')
            image_bytes = buf.getvalue()

        return Response(image_bytes, media_type='image/png')

    except Exception as exception:
        logger.exception(str(exception))
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
