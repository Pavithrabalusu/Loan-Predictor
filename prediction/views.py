from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime
from prediction.ml_model.utils import load_model, load_test_data
from .serializers import (
    LoanPredictionInputSerializer,
    LoanPredictionOutputSerializer,
    LoanPredictionErrorSerializer
)
from django.views.generic import TemplateView
import logging
from django.conf import settings
from rest_framework.permissions import AllowAny
from rest_framework.authentication import SessionAuthentication, BasicAuthentication

logger = logging.getLogger(__name__)

# Constants
BASE_FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education',
    'Self_Employed', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area',
    'ApplicantIncome', 'CoapplicantIncome'
]

# Initialize model and data (will be set in ready())
model = None
test_df = None

def initialize_services():
    """Initialize model and test data with detailed error reporting"""
    global model, test_df
    try:
        logger.info("Attempting to load model...")
        model = load_model()
        logger.info("Model loaded successfully")
        
        logger.info("Attempting to load test data...")
        test_df = load_test_data()
        logger.info("Test data loaded successfully")
        
        # Verify the loaded data
        if test_df.empty:
            raise ValueError("Test DataFrame is empty")
            
        # Verify model can make predictions
        test_sample = test_df.iloc[:1][BASE_FEATURES]
        try:
            _ = model.predict_proba(test_sample)
            logger.info("Model validation successful")
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}", exc_info=True)
        model = None
        test_df = None
        raise  # Re-raise the exception to see it during startup

class PredictLoanView(APIView):
    """
    API endpoint for making loan approval predictions
    """
    authentication_classes = []  # Disable authentication
    permission_classes = [AllowAny]  # Allow all users
    
    def post(self, request):
        # Check if services are ready
        if model is None or test_df is None:
            error_data = {
                'error': 'Prediction service unavailable',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return Response(
                LoanPredictionErrorSerializer(error_data).data,
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Validate input
        serializer = LoanPredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            error_data = {
                'error': 'Invalid input data',
                'details': serializer.errors,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return Response(
                LoanPredictionErrorSerializer(error_data).data,
                status=status.HTTP_400_BAD_REQUEST
            )
            
        loan_id = serializer.validated_data['loan_id']
        
        try:
            # Find loan record
            loan_data = test_df[test_df['Loan_ID'] == loan_id]
            if loan_data.empty:
                error_data = {
                    'error': f'Loan ID {loan_id} not found',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return Response(
                    LoanPredictionErrorSerializer(error_data).data,
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Make prediction
            proba = model.predict_proba(loan_data[BASE_FEATURES])[0][1]
            
            # Prepare response
            result = {
                'loan_id': loan_id,
                'status': "APPROVED" if proba >= 0.5 else "REJECTED",
                'confidence': round(proba * 100, 2),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Prediction made for loan {loan_id}: {result['status']}")
            return Response(
                LoanPredictionOutputSerializer(result).data,
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            logger.error(f"Prediction failed for loan {loan_id}: {str(e)}", exc_info=True)
            error_data = {
                'error': 'Internal prediction error',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return Response(
                LoanPredictionErrorSerializer(error_data).data,
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FrontendView(TemplateView):
    """
    View for serving the frontend interface
    """
    template_name = 'index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['service_available'] = model is not None and test_df is not None
        return context

# Initialize services when Django starts
try:
    initialize_services()
except Exception as e:
    logger.critical(f"Failed to initialize prediction services: {str(e)}")