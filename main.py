import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.churn_api import app as churn_app
from api.return_risk_api import app as return_risk_app
from api.purchase_api import app as purchase_api
from config.database import get_connection
from training.train_churn import main as train_churn
from training.train_return_risk import main as train_return_risk
from training.train_purchase import main as train_purchase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize main FastAPI app
app = FastAPI(
    title="Northwind ML API",
    description="API for customer churn, return risk, and purchase predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers from all APIs
app.include_router(churn_app.router, prefix="/api/v1", tags=["churn"])
app.include_router(return_risk_app.router, prefix="/api/v1", tags=["return-risk"])
app.include_router(purchase_api.router, prefix="/api/v1", tags=["purchase"])

@app.get("/")
async def root():
    """Root endpoint that provides API information."""
    return {
        "name": "Northwind ML API",
        "version": "1.0.0",
        "endpoints": {
            "churn": "/api/v1/predict/churn",
            "return_risk": "/api/v1/predict/return-risk",
            "purchase": "/api/v1/predict/purchase"
        }
    }

def setup_database():
    """Initialize database connection."""
    logger.info("Setting up database connection...")
    try:
        engine = get_connection()
        logger.info("Database connection established successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to establish database connection: {str(e)}")
        raise

def train_models(engine):
    """Train all models in sequence."""
    logger.info("Starting model training pipeline...")
    
    try:
        # Train churn model
        logger.info("Training churn prediction model...")
        train_churn(engine)
        logger.info("Churn model training completed")
        
        # Train return risk model
        logger.info("Training return risk prediction model...")
        train_return_risk(engine)
        logger.info("Return risk model training completed")
        
        # Train purchase model
        logger.info("Training purchase prediction model...")
        train_purchase(engine)
        logger.info("Purchase model training completed")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def start_api():
    """Start the FastAPI application."""
    logger.info("Starting API server...")
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

def main():
    """Main function to run the entire pipeline."""
    try:
        # Setup database
        engine = setup_database()
        
        # Train models
        train_models(engine)
        
        # Start API
        start_api()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
