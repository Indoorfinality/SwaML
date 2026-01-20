from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.shortcuts import render
import os
import tempfile
import uuid
from src.data_utils import load_dataset, preprocess_features
from src.target_detection import detect_target, detect_target_candidates
from src.model_training import train_models
import joblib
from .serializers import (TrainingResultSerializer,
                          ErrorResponseSerializer,
                          UploadFileResponseSerializer,
                          AnalyzeTargetResponseSerializer, PreviewDataSerializer)


UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def home(request):
    """Serve the home page with the upload form"""
    return render(request, 'index.html')



class UploadFile(APIView):
    """
    POST /api/upload-file/
    Uploads a new CSV file and returns filename + basic preview
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        csv_file = request.FILES.get('csv_files')
        if not csv_file:
            return Response({
                "status": "error",
                "error": "No CSV file uploaded",
                "message": "Please provide a CSV file in the 'csv_files' field"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save with unique name
        filename = f"{uuid.uuid4()}_{csv_file.name}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, 'wb+') as dest:
            for chunk in csv_file.chunks():
                dest.write(chunk)
        try:
            df = load_dataset(file_path)
            preview_rows = df.head(10).fillna("N/A").to_dict(orient='records')

            return Response(UploadFileResponseSerializer({
                "message": "File uploaded successfully",
                "uploaded_filename": filename,
                "total_rows": len(df),
                "columns": list(df.columns),
            }).data, status=201)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return Response(ErrorResponseSerializer({
                "error": "Upload failed",
                "message": str(e)
            }).data, status=500)

    

class PreviewDataset(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        """
        Fetch and display rows from an uploaded dataset.
        
        Query Parameters:
            - dataset_name: The filename of the uploaded dataset
            - limit: Number of rows to return (default: 10)
            - offset: Starting row index (default: 0)
        
        Response:
            - Success (200): Returns dataset rows with metadata
            - Error (400/500): Returns error message
        """
        dataset_name = request.GET.get('dataset_name')
        limit = int(request.GET.get('limit', 10))
        offset = int(request.GET.get('offset', 0))

        if not dataset_name:
            return Response({
                "status": "error",
                "message": "No dataset_name provided"
            }, status=400)

        file_path = os.path.join(UPLOAD_DIR, dataset_name)

        if not os.path.exists(file_path):
            return Response({
                "status": "error",
                "message": f"Dataset '{dataset_name}' not found"
            }, status=400)

        try:
            df = load_dataset(file_path)
            total_rows = len(df)

            offset = max(0, offset)
            limit = max(1, limit)
            limit = min(limit, total_rows - offset)
            
            df_slice = df.iloc[offset:offset + limit]
            data_rows = df_slice.fillna("N/A").to_dict(orient='records')

            return Response(PreviewDataSerializer({
                "status": "success",
                "columns": list(df.columns),
                "total_rows": total_rows,
                "offset": offset,
                "limit": limit,
                "returned_rows": len(data_rows),
                "data": data_rows
            }).data, status=200)

        except Exception as e:
            return Response({
                "status": "error",
                "message": str(e)
            }, status=500)

class AnalyzeTarget(APIView):
    """
    POST /api/analyze-target/
    Analyzes a previously uploaded file and returns target suggestion.
    
    Required: uploaded_filename (the filename returned from /api/upload-file/)
    """
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request):
        uploaded_filename = request.data.get('uploaded_filename')
        if not uploaded_filename:
            return Response({
                "error": "No uploaded_filename provided",
                "message": "Please provide an uploaded_filename in the 'uploaded_filename' field"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_filename)
        if not os.path.exists(file_path):
            return Response(ErrorResponseSerializer({
                "error": "File not found",
                "message": f"File '{uploaded_filename}' not found"
            }).data, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = load_dataset(file_path)
            columns = list(df.columns)
            candidates = detect_target_candidates(df)
            target = detect_target(df)
            
            return Response(AnalyzeTargetResponseSerializer({
                "status": "success",
                "detected_target": target,
                "all_columns": columns,
                "candidate_columns": candidates,
                "uploaded_filename": uploaded_filename
            }).data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(ErrorResponseSerializer({
                "status": "error",
                "error": "Analysis failed",
                "message": str(e)
            }).data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class StartTraining(APIView):
    """
    POST /api/start-training/
    Starts training using previously uploaded file + target column
    Required: uploaded_filename, target_column
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        uploaded_filename = request.data.get('uploaded_filename')
        target = request.data.get('target_column')
        
        if not uploaded_filename:
            return Response(ErrorResponseSerializer({
                "error": "No uploaded_filename provided",
                "message": "Please provide an uploaded_filename in the 'uploaded_filename' field"
            }).data, status=status.HTTP_400_BAD_REQUEST)
        
        if not target:
           return Response(ErrorResponseSerializer({
               "error": "No target_column provided",
               "message": "Please provide a target_column in the 'target_column' field"
           }).data, status=status.HTTP_400_BAD_REQUEST)
        
        file_path = os.path.join(UPLOAD_DIR, uploaded_filename)
        if not os.path.exists(file_path):
            return Response(ErrorResponseSerializer({
                "error": "File not found",
                "message": f"File '{uploaded_filename}' not found"
            }).data, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            df = load_dataset(file_path)
            X, y = preprocess_features(df, target)
            best_name, best_model, results = train_models(X, y)

            model_filename = f"model_{uuid.uuid4()}_{best_name.replace(' ', '_')}.pkl"
            model_path = os.path.join(UPLOAD_DIR, model_filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            joblib.dump(best_model, model_path)

            return Response(TrainingResultSerializer({
                "status": "success",
                "best_model": best_name,
                "results": results,
                "model_download_url": f"/media/{model_filename}",
                "feature_plot_url": None
            }).data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(ErrorResponseSerializer({
                "error": "Training failed",
                "message": str(e)
            }).data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



            

