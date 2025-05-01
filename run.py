from config.database import get_connection
from training.train_churn import main as train_churn
from training.train_return_risk import main as train_return_risk
from training.train_purchase import main as train_purchase

def main():
    """
    Main function to run all training scripts.
    """
    try:
        # Get database connection
        engine = get_connection()
        
        # Train models
        train_churn(engine)
        train_return_risk(engine)
        train_purchase(engine)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 