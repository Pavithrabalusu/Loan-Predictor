import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
import os
from pathlib import Path

# Set up paths - adjust these based on your deployment structure
MODEL_DIR = Path(__file__).parent.parent / "prediction" / "ml_model" / "saved_models"
MODEL_PATHS = {
    'pipeline': MODEL_DIR / 'loan_model.pkl',  # Note .pkl extension
    'xgb_model': MODEL_DIR / 'loan_model.json'
}

# Debugging - add these lines temporarily
print(f"Current directory: {Path.cwd()}")
print(f"Looking for model at: {MODEL_PATHS['pipeline']}")
print(f"File exists: {MODEL_PATHS['pipeline'].exists()}")

# Custom load function adapted from your utils.py
@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    try:
        # Load the complete pipeline if possible
        try:
            pipeline = joblib.load(MODEL_PATHS['pipeline'])
            if hasattr(pipeline, 'predict_proba'):
                return pipeline
        except Exception as e:
            st.warning(f"Full pipeline load failed: {str(e)}")
        
        # Fallback: Reconstruct pipeline
        # 1. Load the preprocessor from the original pipeline
        original_pipeline = joblib.load(MODEL_PATHS['pipeline'])
        preprocessor = original_pipeline.named_steps['preprocessor']
        
        # 2. Create a new XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        
        # 3. Load the booster weights
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATHS['xgb_model']))
        xgb_model._Booster = booster
        
        # 4. Set required sklearn attributes
        xgb_model._estimator_type = "classifier"
        xgb_model.classes_ = [0, 1]
        xgb_model.n_classes_ = 2
        
        # 5. Reconstruct the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb_model)
        ])
        
        return pipeline
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

# Feature engineering function adapted from loan_predictor.py
def add_features(df):
    """Feature engineering"""
    df = df.copy()
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term'].replace(0, 1)
    df['IncomeToLoanRatio'] = df['TotalIncome'] / df['LoanAmount'].replace(0, 1)
    return df

# Main app
def main():
    st.title("Loan Approval Predictor")
    st.write("This app predicts whether a loan application will be approved based on applicant information.")
    
    # Load model
    try:
        model = load_model()
    except Exception as e:
        st.error("Failed to load prediction model. Please check the model files.")
        st.stop()
    
    # Create input form
    with st.form("loan_input_form"):
        st.header("Applicant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_amount_term = st.number_input("Loan Term (months)", min_value=0)
            credit_history = st.selectbox("Credit History", ["1 (Good)", "0 (Bad)"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        submitted = st.form_submit_button("Predict Loan Approval")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': 1 if credit_history.startswith("1") else 0,
            'Property_Area': property_area
        }])
        
        # Add engineered features
        input_data = add_features(input_data)
        
        # Make prediction
        try:
            proba = model.predict_proba(input_data)[0][1]
            prediction = "APPROVED" if proba >= 0.5 else "REJECTED"
            confidence = round(proba * 100, 2) if prediction == "APPROVED" else round((1 - proba) * 100, 2)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction == "APPROVED":
                st.success(f"✅ Loan Approved with {confidence}% confidence")
            else:
                st.error(f"❌ Loan Rejected with {confidence}% confidence")
            
            # Show probability breakdown
            st.write("Probability Breakdown:")
            prob_df = pd.DataFrame({
                'Status': ['Approved', 'Rejected'],
                'Probability': [f"{proba*100:.2f}%", f"{(1-proba)*100:.2f}%"]
            })
            st.table(prob_df)
            
            # Show feature importance (if available)
            try:
                if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                    st.subheader("Key Decision Factors")
                    importances = model.named_steps['classifier'].feature_importances_
                    features = model.named_steps['preprocessor'].get_feature_names_out()
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
            except Exception:
                pass
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()