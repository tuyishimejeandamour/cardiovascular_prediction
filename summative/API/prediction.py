from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
import joblib
import json
import pandas as pd
from typing import List, Dict, Any

# Define global variables for model and scaler
model = None
scaler = None

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and scaler on startup
    global model, scaler
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        # This will be caught when the API is accessed
        model = None
        scaler = None
    
    yield
    
    # Clean up resources on shutdown if needed
    # No cleanup needed for this application

# Create FastAPI app
app = FastAPI(
    title="Cardiovascular Disease Risk Score API",
    description="API for calculating a cardiovascular disease risk score based on health metrics. The risk score is a continuous value between 0 and 1, where higher values indicate higher risk.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check endpoint to verify API status"
        },
        {
            "name": "Prediction",
            "description": "Endpoints for making cardiovascular disease risk predictions"
        },
        {
            "name": "Guides",
            "description": "Endpoints for retrieving health recommendations based on risk scores"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load model info
try:
    with open("models/model_info.json", "r") as f:
        model_info = json.load(f)
    features = model_info["features"]
    model_name = model_info["model_name"]
except FileNotFoundError:
    features = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    model_name = "Unknown"

# Load numerical features list
try:
    with open("models/numerical_features.json", "r") as f:
        numerical_features = json.load(f)
except FileNotFoundError:
    numerical_features = ["age", "height", "weight", "ap_hi", "ap_lo"]

# Define input data model with constraints
class CardiovascularPredictionInput(BaseModel):
    age: int = Field(
        ..., 
        ge=7300,  # 20 years in days
        le=36500,  # 100 years in days
        description="Age in days (7300-36500)"
    )
    gender: int = Field(
        ..., 
        ge=1, 
        le=2, 
        description="Gender (1=female, 2=male)"
    )
    height: int = Field(
        ..., 
        ge=120, 
        le=220, 
        description="Height in cm (120-220)"
    )
    weight: float = Field(
        ..., 
        ge=40, 
        le=200, 
        description="Weight in kg (40-200)"
    )
    ap_hi: int = Field(
        ..., 
        ge=80, 
        le=240, 
        description="Systolic blood pressure (80-240 mmHg)"
    )
    ap_lo: int = Field(
        ..., 
        ge=40, 
        le=160, 
        description="Diastolic blood pressure (40-160 mmHg)"
    )
    cholesterol: int = Field(
        ..., 
        ge=1, 
        le=3, 
        description="Cholesterol level (1=normal, 2=above normal, 3=well above normal)"
    )
    gluc: int = Field(
        ..., 
        ge=1, 
        le=3, 
        description="Glucose level (1=normal, 2=above normal, 3=well above normal)"
    )
    smoke: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Smoking status (0=non-smoker, 1=smoker)"
    )
    alco: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Alcohol intake (0=doesn't drink, 1=drinks)"
    )
    active: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Physical activity (0=not active, 1=active)"
    )

    # Additional validators
    @field_validator('smoke', 'alco', 'active', mode='before')
    @classmethod
    def must_be_binary(cls, v, info):
        if v not in [0, 1]:
            raise ValueError(f'{info.field_name} must be either 0 or 1')
        return v
    
    @field_validator('cholesterol', 'gluc', mode='before')
    @classmethod
    def must_be_valid_level(cls, v, info):
        if v not in [1, 2, 3]:
            raise ValueError(f'{info.field_name} must be 1, 2, or 3')
        return v
    
    @field_validator('ap_lo', mode='before')
    @classmethod
    def diastolic_lower_than_systolic(cls, v, info):
        if 'ap_hi' in info.data and v >= info.data['ap_hi']:
            raise ValueError('Diastolic blood pressure (ap_lo) must be lower than systolic blood pressure (ap_hi)')
        return v

class PredictionResponse(BaseModel):
    risk_score: float
    model_name: str
    input_data: Dict[str, Any]

class GuideRequest(BaseModel):
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk score between 0 and 1"
    )
    
    @field_validator('risk_score', mode='before')
    @classmethod
    def validate_risk_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Risk score must be between 0 and 1")
        return v

class Recommendation(BaseModel):
    title: str
    description: str
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Priority level from 1 (highest) to 5 (lowest)"
    )

class GuideResponse(BaseModel):
    risk_category: str
    risk_score: float
    general_advice: str
    recommendations: List[Recommendation]

# Model and scaler are now loaded in the lifespan context manager

# Health check endpoint
@app.get("/", 
    tags=["Health"],
    summary="API Health Check",
    description="Returns the status of the API and the loaded model name.",
    response_description="API status information"
)
async def root():
    """
    Health check endpoint to verify the API is running and which model is loaded.
    
    Returns:
        dict: A dictionary containing the API status and model name
    """
    return {"status": "API is running", "model": model_name}

# Prediction endpoint
@app.post(
    "/predict", 
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Cardiovascular Disease Risk",
    description="Calculates a cardiovascular disease risk score based on the provided health metrics. The risk score is a continuous value between 0 and 1, where higher values indicate higher risk.",
    response_description="Risk score prediction result with input data",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "risk_score": 0.42,
                        "model_name": "RandomForestClassifier",
                        "input_data": {
                            "age": 18250,
                            "gender": 2,
                            "height": 175,
                            "weight": 80,
                            "ap_hi": 120,
                            "ap_lo": 80,
                            "cholesterol": 1,
                            "gluc": 1,
                            "smoke": 0,
                            "alco": 0,
                            "active": 1
                        }
                    }
                }
            }
        },
        500: {
            "description": "Model not loaded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model or scaler not loaded. Please run the notebook first to train and save the model."
                    }
                }
            }
        }
    }
)
async def predict(input_data: CardiovascularPredictionInput):
    """
    Predicts the cardiovascular disease risk based on the provided health metrics.
    
    The prediction is made using a machine learning model trained on cardiovascular disease data.
    The input data is first scaled using a pre-trained scaler, then passed to the model for prediction.
    
    Args:
        input_data (CardiovascularPredictionInput): Health metrics data
        
    Returns:
        PredictionResponse: Risk score prediction result with input data
        
    Raises:
        HTTPException: If the model or scaler is not loaded
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500, 
            detail="Model or scaler not loaded. Please run the notebook first to train and save the model."
        )
    
    # Convert input data to dictionary
    input_dict = input_data.model_dump()
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_dict])
    
    # Scale the numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Make prediction
    risk_score = float(model.predict(input_df)[0])
    
    # Ensure risk score is between 0 and 1 for interpretability
    risk_score = max(0, min(1, risk_score))
    
    # Return risk score and input data
    return PredictionResponse(
        risk_score=risk_score,
        model_name=model_name,
        input_data=input_dict
    )

# Guides endpoint
@app.post(
    "/guides", 
    response_model=GuideResponse,
    tags=["Guides"],
    summary="Get Health Recommendations Based on Risk Score",
    description="Provides personalized health recommendations and guidance based on the cardiovascular disease risk score. Different recommendations are provided for low, moderate, and high risk categories.",
    response_description="Health recommendations and guidance based on risk score",
    responses={
        200: {
            "description": "Successful guide generation",
            "content": {
                "application/json": {
                    "example": {
                        "risk_category": "Moderate",
                        "risk_score": 0.42,
                        "general_advice": "Your cardiovascular risk is moderate. While not immediately concerning, you should take steps to improve your heart health.",
                        "recommendations": [
                            {
                                "title": "Regular Blood Pressure Monitoring",
                                "description": "Check your blood pressure at least once a month and keep a log of readings.",
                                "priority": 2
                            },
                            {
                                "title": "Dietary Adjustments",
                                "description": "Reduce sodium intake and increase consumption of fruits, vegetables, and whole grains.",
                                "priority": 1
                            },
                            {
                                "title": "Regular Exercise",
                                "description": "Aim for at least 150 minutes of moderate-intensity exercise per week.",
                                "priority": 1
                            }
                        ]
                    }
                }
            }
        },
        400: {
            "description": "Invalid risk score",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Risk score must be between 0 and 1"
                    }
                }
            }
        }
    }
)
async def get_guides(guide_request: GuideRequest):
    """
    Provides personalized health recommendations based on the cardiovascular disease risk score.
    
    The recommendations are categorized into three risk levels:
    - Low Risk (0-0.3): General preventive measures and healthy lifestyle maintenance
    - Moderate Risk (0.3-0.7): Targeted lifestyle modifications and regular monitoring
    - High Risk (0.7-1.0): Urgent interventions and medical consultation recommendations
    
    Args:
        guide_request (GuideRequest): Request containing the risk score
        
    Returns:
        GuideResponse: Personalized health recommendations and guidance
        
    Raises:
        HTTPException: If the risk score is invalid
    """
    risk_score = guide_request.risk_score
    
    # Determine risk category
    if risk_score < 0.3:
        risk_category = "Low"
        general_advice = "Your cardiovascular risk is low. Continue maintaining a healthy lifestyle to keep it that way."
        recommendations = [
            Recommendation(
                title="Regular Health Check-ups",
                description="Schedule a general health check-up once a year to monitor your cardiovascular health.",
                priority=3
            ),
            Recommendation(
                title="Balanced Diet",
                description="Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
                priority=2
            ),
            Recommendation(
                title="Regular Physical Activity",
                description="Continue with at least 150 minutes of moderate exercise per week.",
                priority=2
            ),
            Recommendation(
                title="Limit Alcohol Consumption",
                description="If you drink alcohol, do so in moderation (up to one drink per day for women and up to two drinks per day for men).",
                priority=4
            )
        ]
    elif risk_score < 0.7:
        risk_category = "Moderate"
        general_advice = "Your cardiovascular risk is moderate. While not immediately concerning, you should take steps to improve your heart health."
        recommendations = [
            Recommendation(
                title="Regular Blood Pressure Monitoring",
                description="Check your blood pressure at least once a month and keep a log of readings.",
                priority=2
            ),
            Recommendation(
                title="Dietary Adjustments",
                description="Reduce sodium intake and increase consumption of fruits, vegetables, and whole grains.",
                priority=1
            ),
            Recommendation(
                title="Regular Exercise",
                description="Aim for at least 150 minutes of moderate-intensity exercise per week.",
                priority=1
            ),
            Recommendation(
                title="Stress Management",
                description="Practice stress-reduction techniques such as meditation, deep breathing, or yoga.",
                priority=3
            ),
            Recommendation(
                title="Regular Health Check-ups",
                description="Schedule a check-up with your healthcare provider every 6 months.",
                priority=2
            )
        ]
    else:
        risk_category = "High"
        general_advice = "Your cardiovascular risk is high. It's important to take immediate steps to reduce your risk and consult with a healthcare professional."
        recommendations = [
            Recommendation(
                title="Consult a Healthcare Professional",
                description="Schedule an appointment with a cardiologist or primary care physician as soon as possible.",
                priority=1
            ),
            Recommendation(
                title="Blood Pressure Management",
                description="Monitor your blood pressure daily and follow your doctor's recommendations for management.",
                priority=1
            ),
            Recommendation(
                title="Medication Adherence",
                description="If prescribed medication, take it exactly as directed by your healthcare provider.",
                priority=1
            ),
            Recommendation(
                title="Dietary Changes",
                description="Follow a heart-healthy diet low in sodium, saturated fats, and added sugars. Consider consulting with a nutritionist.",
                priority=2
            ),
            Recommendation(
                title="Physical Activity Plan",
                description="Develop a supervised exercise plan with guidance from a healthcare professional.",
                priority=2
            ),
            Recommendation(
                title="Smoking Cessation",
                description="If you smoke, seek help to quit immediately. Consider nicotine replacement therapy or other cessation programs.",
                priority=1
            )
        ]
    
    return GuideResponse(
        risk_category=risk_category,
        risk_score=risk_score,
        general_advice=general_advice,
        recommendations=recommendations
    )

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)