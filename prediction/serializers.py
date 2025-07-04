from rest_framework import serializers
from datetime import datetime
import re
from django.utils.translation import gettext_lazy as _

class LoanPredictionInputSerializer(serializers.Serializer):
    """
    Serializer for loan prediction input validation
    Validates loan ID format: alphanumeric with optional hyphens/underscores
    """
    loan_id = serializers.CharField(
        max_length=100,
        trim_whitespace=True,
        required=True,
        help_text="Unique loan identifier (alphanumeric with hyphens/underscores)"
    )

    def validate_loan_id(self, value):
        """Custom validation for loan ID format"""
        if not re.match(r'^[A-Za-z0-9_-]+$', value):
            raise serializers.ValidationError(
                _("Loan ID must contain only letters, numbers, hyphens and underscores")
            )
        return value.strip()

class LoanPredictionOutputSerializer(serializers.Serializer):
    """
    Serializer for standardized prediction output
    Includes status, confidence score and timestamp
    """
    loan_id = serializers.CharField(
        max_length=100,
        read_only=True,
        help_text="Unique loan identifier"
    )
    status = serializers.ChoiceField(
        choices=[('APPROVED', 'Approved'), ('REJECTED', 'Rejected')],
        read_only=True,
        help_text="Loan approval status"
    )
    confidence = serializers.FloatField(
        min_value=0,
        max_value=100,
        read_only=True,
        help_text="Prediction confidence score (0-100)"
    )
    timestamp = serializers.DateTimeField(
        read_only=True,
        format="%Y-%m-%d %H:%M:%S",
        help_text="Prediction timestamp in UTC"
    )

    def to_representation(self, instance):
        """Format the output consistently"""
        data = super().to_representation(instance)
        # Ensure confidence is rounded to 2 decimal places
        data['confidence'] = round(data['confidence'], 2)
        return data

class LoanPredictionErrorSerializer(serializers.Serializer):
    """
    Serializer for error responses
    """
    error = serializers.CharField(
        max_length=200,
        help_text="Error message description"
    )
    details = serializers.DictField(
        required=False,
        help_text="Additional error details"
    )
    timestamp = serializers.DateTimeField(
        read_only=True,
        format="%Y-%m-%d %H:%M:%S",
        help_text="Error timestamp in UTC"
    )