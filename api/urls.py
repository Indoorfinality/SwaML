from django.urls import path
from .views import UploadAndTrain

urlpatterns = [
    path('upload/', UploadAndTrain.as_view(), name='upload-and-train'),
]

