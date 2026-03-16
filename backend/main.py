from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import io
from pathlib import Path

# Add the parent directory to Python path so we can import from `training`
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import the prediction module we just built
try:
    from training.predict import predict_disease
except ImportError:
    print("Warning: Could not import predict_disease. Make sure the file exists.")
    
app = FastAPI(title="AI-Powered Plant Disease Detection API")

# Step 7 Requirement: Configure CORS so the frontend can easily communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, change this to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    status: str
    diseaseName: str
    confidenceScore: float
    confidencePercent: str
    advice: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FlorAI Plant Disease Detection API!"}

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict_plant_disease(file: UploadFile = File(...)):
    """
    Endpoint mapping to receive an uploaded leaf image and return a real model prediction.
    """
    try:
        # Read the uploaded image payload as bytes
        image_data = await file.read()
        image_stream = io.BytesIO(image_data)
        
        # Step 6 Integration: Call our prediction module
        # The predict_disease function pre-processes the image, runs the CNN, 
        # and formats the output into a dictionary perfectly matching our PredictionResponse model.
        result_dict = predict_disease(image_stream)
        
        # If there's an error from predict_disease, format it nicely
        if "error" in result_dict:
            return PredictionResponse(
                status="Error",
                diseaseName="Prediction Failed",
                confidenceScore=0.0,
                confidencePercent="0%",
                advice=f"Error executing prediction model: {result_dict['error']}"
            )
            
        return PredictionResponse(**result_dict)
        
    except Exception as e:
        return PredictionResponse(
            status="Error",
            diseaseName="Processing Failed",
            confidenceScore=0.0,
            confidencePercent="0%",
            advice=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
