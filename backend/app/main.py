
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="Smart PPE Detection API",
    description="AI-powered Personal Protective Equipment Detection System",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart PPE Detection API",
        "project": "Smart PPE Detection",
        "status": "Running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/metadata")
async def model_metadata():
    return {
        "model_name": "YOLO",
        "model_version": "Not trained yet",
        "framework": "Ultralytics",
        "status": "Not loaded"
    }
