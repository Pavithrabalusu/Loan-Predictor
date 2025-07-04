import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Constants - using absolute paths
DATA_DIR = SCRIPT_DIR  # Files are in the same directory as the script
MODEL_DIR = SCRIPT_DIR / "saved_models"  # Create a subdirectory for models
os.makedirs(MODEL_DIR, exist_ok=True)

class LoanPredictor:
    def __init__(self):
        self.base_features = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area',
            'ApplicantIncome', 'CoapplicantIncome'
        ]
        self.numeric_features = [
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'ApplicantIncome', 'CoapplicantIncome'
        ]
        self.categorical_features = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area'
        ]
        self.model = None
        self.predictions_log = []
        
    def add_features(self, df):
        """Feature engineering"""
        df = df.copy()
        df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term'].replace(0, 1)  # Avoid division by zero
        df['IncomeToLoanRatio'] = df['TotalIncome'] / df['LoanAmount'].replace(0, 1)
        return df

    def build_pipeline(self):
        """Build the model pipeline"""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])

        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                use_label_encoder=False
            ))
        ])

    def load_data(self):
        """Load and preprocess data"""
        try:
            train_path = DATA_DIR / 'train_u6lujuX_CVtuZ9i.csv'
            test_path = DATA_DIR / 'test_Y3wMUE5_7gLdaTN.csv'
            
            print(f"Loading training data from: {train_path}")
            print(f"Loading test data from: {test_path}")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            return self.add_features(train_df), self.add_features(test_df)
            
        except Exception as e:
            print(f"\nError loading data: {e}")
            print("Please ensure both CSV files are in the same directory as the script:")
            print(f"Directory: {DATA_DIR}")
            print("Required files:")
            print("- train_u6lujuX_CVtuZ9i.csv")
            print("- test_Y3wMUE5_TgLdaTN.csv\n")
            exit(1)

    def train_model(self):
        """Train and save the model"""
        train_df, _ = self.load_data()
        X_train = train_df[self.base_features]
        y_train = train_df['Loan_Status'].map({'Y': 1, 'N': 0})

        self.model = self.build_pipeline()
        print("\nTraining model...")
        self.model.fit(X_train, y_train)

        # Verify training
        try:
            check_is_fitted(self.model)
            train_preds = self.model.predict(X_train)
            print(f"✓ Model trained successfully! Training accuracy: {accuracy_score(y_train, train_preds):.2%}")
            
            # Save models
            self.save_models()
        except Exception as e:
            print(f"! Training failed: {e}")
            exit(1)

    def save_models(self):
        """Save models with all required attributes"""
        # Ensure the classifier has required attributes
        classifier = self.model.named_steps['classifier']
        if not hasattr(classifier, 'classes_'):
            classifier.classes_ = [0, 1]
            classifier.n_classes_ = 2
            classifier._estimator_type = "classifier"
        
        # Save complete pipeline
        model_path = MODEL_DIR / 'loan_model.pkl'
        joblib.dump(self.model, model_path)
        
        # Save XGBoost model separately
        xgb_path = MODEL_DIR / 'loan_model.json'
        classifier.save_model(str(xgb_path))
        
        print(f"✓ Models saved to: {MODEL_DIR}")

    def predict_loan(self, loan_id, test_df):
        """Make prediction for a single loan"""
        loan_data = test_df[test_df['Loan_ID'] == loan_id]
        if loan_data.empty:
            print(f"Loan ID {loan_id} not found!")
            return None

        proba = self.model.predict_proba(loan_data[self.base_features])[0][1]
        status = "APPROVED" if proba >= 0.5 else "REJECTED"
        confidence = round(proba * 100, 2)

        result = {
            'Loan_ID': loan_id,
            'Status': status,
            'Confidence': confidence,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.predictions_log.append(result)
        return result

    def save_predictions(self):
        """Save prediction log"""
        if not self.predictions_log:
            print("\nNo predictions to save!")
            return

        master_file = DATA_DIR / "all_predictions.csv"
        try:
            existing_df = pd.read_csv(master_file) if master_file.exists() else None
            new_df = pd.DataFrame(self.predictions_log)
            combined_df = pd.concat([existing_df, new_df]) if existing_df is not None else new_df
            combined_df.to_csv(master_file, index=False)
            print(f"\nPredictions saved to {master_file}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
        
        self.predictions_log.clear()

    def interactive_predict(self):
        """Interactive prediction interface"""
        _, test_df = self.load_data()
        
        try:
            while True:
                print("\n" + "="*50)
                loan_id = input("Enter Loan ID (or 'quit' to exit): ").strip()

                if loan_id.lower() == 'quit':
                    if self.predictions_log:
                        self.save_predictions()
                    print("Exiting program...")
                    break

                result = self.predict_loan(loan_id, test_df)
                if result:
                    print("\n" + "="*50)
                    print(f"Loan ID: {result['Loan_ID']}")
                    print(f"Status: {result['Status']}")
                    print(f"Confidence: {result['Confidence']}%")
                    print("="*50 + "\n")
                    
        except KeyboardInterrupt:
            if self.predictions_log:
                self.save_predictions()
            print("\nProgram interrupted. Predictions saved.")

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {SCRIPT_DIR}")
    
    predictor = LoanPredictor()
    predictor.train_model()
    predictor.interactive_predict()