import joblib
import pandas as pd
import os
import xgboost as xgb
from django.conf import settings
import logging
from sklearn.preprocessing import LabelEncoder
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path configuration
MODEL_DIR = os.path.join(settings.BASE_DIR, 'prediction', 'ml_model')
# In utils.py, update the MODEL_PATHS:
MODEL_PATHS = {
    'pipeline': os.path.join(MODEL_DIR, 'saved_models', 'loan_model.pkl'),  # Updated path
    'xgb_model': os.path.join(MODEL_DIR, 'saved_models', 'loan_model.json'),  # Updated path
    'test_data': os.path.join(MODEL_DIR, 'test_Y3wMUE5_7gLdaTN.csv')
}

def _validate_model_files():
    """Verify all required model files exist"""
    missing_files = [name for name, path in MODEL_PATHS.items() if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {', '.join(missing_files)}. "
            f"Please ensure these files exist in {MODEL_DIR}"
        )

def load_model():
    """Load the trained model pipeline with proper sklearn compatibility"""
    try:
        _validate_model_files()
        
        # Load the complete pipeline if possible
        try:
            pipeline = joblib.load(MODEL_PATHS['pipeline'])
            if hasattr(pipeline, 'predict_proba'):
                logger.info("Loaded complete pipeline successfully")
                return pipeline
        except Exception as e:
            logger.warning(f"Full pipeline load failed: {str(e)}")
        
        # Fallback: Reconstruct pipeline with proper sklearn compatibility
        logger.info("Reconstructing pipeline with XGBoost model")
        
        # 1. Load the preprocessor from the original pipeline
        original_pipeline = joblib.load(MODEL_PATHS['pipeline'])
        preprocessor = original_pipeline.named_steps['preprocessor']
        
        # 2. Create a new XGBoost classifier with proper sklearn tags
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        
        # 3. Load the booster weights
        booster = xgb.Booster()
        booster.load_model(MODEL_PATHS['xgb_model'])
        xgb_model._Booster = booster
        
        # 4. Manually set required sklearn attributes
        xgb_model._estimator_type = "classifier"
        xgb_model.classes_ = [0, 1]
        xgb_model.n_classes_ = 2
        
        # 5. Reconstruct the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_model)
        ])
        
        logger.info("Pipeline reconstructed successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load model: {str(e)}")

def load_test_data():
    """
    Load and preprocess test data with proper validation
    Returns:
        DataFrame: Processed test data
    Raises:
        ValueError: If data loading or processing fails
    """
    try:
        if not os.path.exists(MODEL_PATHS['test_data']):
            raise FileNotFoundError(f"Test data not found at {MODEL_PATHS['test_data']}")
        
        df = pd.read_csv(MODEL_PATHS['test_data'])
        
        # Validate required columns
        required_columns = [
            'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Test data missing required columns: {missing_cols}")
        
        # Feature engineering with null checks
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
        
        numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for col in numeric_cols:
            if df[col].isnull().any():
                logger.warning(f"Column {col} contains null values, filling with median")
                df[col] = df[col].fillna(df[col].median())
        
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term'].replace(0, 1)  # Avoid division by zero
        df['IncomeToLoanRatio'] = df['TotalIncome'] / df['LoanAmount'].replace(0, 1)
        
        logger.info("Test data loaded and processed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Test data loading failed: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load test data: {str(e)}")