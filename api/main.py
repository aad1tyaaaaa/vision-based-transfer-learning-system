from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np

app = FastAPI(title="Vision-Based Transfer Learning API", description="API for image classification using transfer learning model")

# Load TFLite model
interpreter = None
try:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path='models/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")

CLASSES = ['cat', 'dog']

def preprocess_image(image):
    """
    Preprocess image for model input.
    """
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    """
    if interpreter is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_data = preprocess_image(image)
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = output_data[0]
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        
        return JSONResponse(content={
            "class": CLASSES[predicted_class],
            "confidence": confidence,
            "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        }, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.get("/")
async def root():
    return {"message": "Vision-Based Transfer Learning API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)