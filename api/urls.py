from django.urls import path
from .views import UploadAndTrain, DetectTarget

urlpatterns = [
    path('detect-target/', DetectTarget.as_view(), name='detect-target'),
    path('upload/', UploadAndTrain.as_view(), name='upload-and-train'),
]

