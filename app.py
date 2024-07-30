from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
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
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Preprocess the audio
    input_tensor = preprocess_audio(file_path)

    # Debug: Print input tensor shape
    print(f"Input tensor shape: {input_tensor.shape}")

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)

        # Debug: Print raw prediction logits
        print(f"Raw prediction logits: {prediction}")

        predicted_emotion = torch.argmax(prediction, dim=1).item()

        # Debug: Print predicted emotion index
        print(f"Predicted emotion index: {predicted_emotion}")

    predicted_emotion_label = emotion_labels[predicted_emotion]
    return JSONResponse(content={'emotion': predicted_emotion_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
