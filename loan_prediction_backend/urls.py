from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from prediction.views import FrontendView

urlpatterns = [
    # Frontend
    path('', FrontendView.as_view(), name='frontend'),
    
    # API endpoints
    path('api/', include('prediction.urls')),
    
    # Admin (with custom URL from settings)
    path(settings.ADMIN_URL, admin.site.urls),
]

# Static/media files for development only
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)