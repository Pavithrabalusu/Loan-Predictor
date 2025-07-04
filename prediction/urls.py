from django.conf import settings  # Add this import
from django.urls import path
from .views import PredictLoanView

urlpatterns = [
    path('predict/', PredictLoanView.as_view(), name='predict-loan'),
]

# Conditional imports for development
if settings.DEBUG:
    from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
    urlpatterns += [
        path('schema/', SpectacularAPIView.as_view(), name='schema'),
        path('docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    ]