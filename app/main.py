import io
import os
import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from ultralytics import YOLO

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI App Initialization
# ---------------------------
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection using the YOLOv8 model.",
    version="1.0"
)

# ---------------------------
# Model Initialization
# ---------------------------
MODEL_PATH = "D:/All Projects/Interview Assignments/iotech/yolo_best.pt"
logger.info(f"Loading YOLOv8 model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load YOLOv8 model from {MODEL_PATH}: {e}")
    raise RuntimeError(f"Failed to load YOLOv8 model from {MODEL_PATH}: {e}")

# ---------------------------
# Helper Function: Process Image
# ---------------------------
def read_imagefile(file: bytes) -> np.ndarray:
    """
    Read an uploaded image file and convert it into a NumPy array.
    """
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        logger.info("Image file successfully converted to NumPy array.")
        return np.array(image)
    except Exception as e:
        logger.exception(f"Error processing the image file: {e}")
        raise ValueError(f"Could not process the image file: {e}")

# ---------------------------
# Endpoint: /detect
# ---------------------------
@app.post("/detect", summary="Detect objects in an image")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image file and receive JSON detection results.
    
    Returns:
      - List of detections with label, confidence, and bounding box coordinates.
    """
    logger.info(f"Received file: {file.filename}")
    allowed_extensions = {"jpg", "jpeg", "png", "bmp", "gif", "heic"}
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_extensions:
        logger.error(f"File extension .{ext} not allowed.")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed types: jpg, jpeg, png, bmp, gif, heic."
        )
    
    try:
        contents = await file.read()
        image_np = read_imagefile(contents)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Starting inference on the received image...")
    try:
        # Run inference on the image
        results = model(image_np)
        result = results[0]  # Since we're processing one image
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confs = result.boxes.conf.cpu().numpy()    # Confidence scores
            cls_ids = result.boxes.cls.cpu().numpy()   # Class IDs

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                label = (
                    model.model.names[int(cls_id)]
                    if hasattr(model.model, "names")
                    else str(int(cls_id))
                )
                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": box.tolist()  # [x1, y1, x2, y2]
                })
            logger.info(f"Inference completed with {len(detections)} detections.")
        else:
            logger.info("Inference completed but no objects were detected.")
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
    return JSONResponse(content={"detections": detections})

# ---------------------------
# Endpoint: /detect_image
# ---------------------------
@app.post("/detect_image", summary="Detect objects and return an annotated image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload an image file and receive an image with detected objects annotated.
    
    Returns:
      - Annotated JPEG image.
    """
    logger.info(f"Received file for image annotation: {file.filename}")
    allowed_extensions = {"jpg", "jpeg", "png", "bmp", "gif", "heic"}
    ext = file.filename.split(".")[-1].lower()
    if ext not in allowed_extensions:
        logger.error(f"File extension .{ext} not allowed for annotated image.")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed types: jpg, jpeg, png, bmp, gif, heic."
        )
    
    try:
        contents = await file.read()
        image_np = read_imagefile(contents)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    logger.info("Starting inference for annotated image...")
    try:
        results = model(image_np)
        result = results[0]
        # The plot() method returns an annotated image as a NumPy array (BGR format)
        annotated_image = result.plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if not is_success:
            logger.error("Could not encode the annotated image.")
            raise Exception("Could not encode the annotated image.")
        logger.info("Annotated image created successfully.")
    except Exception as e:
        logger.exception(f"Inference error while creating annotated image: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
