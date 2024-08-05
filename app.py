from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import traceback
from utils import preprocess_audio, load_model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model_checkpoint = 'trained_models/RAVDESS/RAVDESS-model-fold-1-epoch=149-train_loss=0.1548.ckpt'
model = load_model(model_checkpoint)
model.eval()

emotion_labels = ["calm", "neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

@app.post("/predict")
async def predict(file: UploadFile = File(None)):
    try:
        # Handle file upload
        if file is None or file.filename == '':
            raise HTTPException(status_code=400, detail="No file provided or file is empty")

        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        print(f"File saved: {file_path}")

        # Preprocess the audio file
        input_tensor = preprocess_audio(file_path)
        print(f"Input tensor shape after preprocessing: {input_tensor.shape}")

        # Ensure input_tensor is 4D: [batch_size, channels, height, width]
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(1)  # Add channel dimension

        print(f"Final input tensor shape: {input_tensor.shape}")

        # Move input tensor to the same device as the model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Model prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            print(f"Raw model output: {prediction}")

            # Assuming the prediction is a tensor with dimensions like [batch_size, num_classes]
            # Get the max value across the class dimension to find the predicted class
            max_values, predicted_classes = torch.max(prediction, dim=1)

        # Convert tensor to list for JSON response
        predicted_classes_list = predicted_classes.tolist()
        predicted_emotion_labels = [emotion_labels[pred] for pred in predicted_classes_list]

        print(f"Predicted emotion labels: {predicted_emotion_labels}")

        return JSONResponse(content={'emotion': predicted_emotion_labels})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
