from django.urls import path
from .views import UploadFile, PreviewDataset, AnalyzeTarget, StartTraining

urlpatterns = [
    path('upload/', UploadFile.as_view(), name='upload-file'),
    path('preview/', PreviewDataset.as_view(), name='preview-dataset'),
    path('analyze-target/', AnalyzeTarget.as_view(), name='analyze-target'),
    path('start-training/', StartTraining.as_view(), name='start-training'),

]

