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
from .serializers import TrainingResultSerializer


def home(request):
    """Serve the home page with the upload form"""
    return render(request, 'index.html')


class DetectTarget(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        """
        Detect target column from uploaded CSV file.
        
        Request:
            - Method: POST
            - Content-Type: multipart/form-data
            - Body: FormData with field 'csv_files' containing the CSV file
        
        Response:
            - Success (200): Returns detected target and all columns
            - Error (400/500): Returns error message
        """
        csv_file = request.FILES.get('csv_files')
        if not csv_file:
            return Response({
                "status": "error",
                "error": "No CSV file uploaded",
                "message": "Please provide a CSV file in the 'csv_files' field"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.csv")

        with open(file_path, 'wb+') as dest:
            for chunk in csv_file.chunks():
                dest.write(chunk)

        try:
            df = load_dataset(file_path)
            columns = list(df.columns)
            candidates = detect_target_candidates(df)
            
            # Detect target without confirmation
            target = detect_target(df)
            
            os.remove(file_path)
            os.rmdir(temp_dir)

            return Response({
                "status": "success",
                "detected_target": target,
                "all_columns": columns,
                "candidate_columns": candidates
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # Cleanup temp files
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            
            return Response({
                "status": "error",
                "error": str(e),
                "message": "An error occurred during target detection. Please check your CSV file format."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UploadAndTrain(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        """
        Upload CSV file and train ML models automatically.
        
        Request:
            - Method: POST
            - Content-Type: multipart/form-data
            - Body: FormData with field 'csv_files' containing the CSV file
        
        Response:
            - Success (201): Returns training results with model info
            - Error (400/500): Returns error message
        """
        csv_file = request.FILES.get('csv_files')
        if not csv_file:
            return Response({
                "status": "error",
                "error": "No CSV file uploaded",
                "message": "Please provide a CSV file in the 'csv_files' field"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.csv")

        with open(file_path, 'wb+') as dest:
            for chunk in csv_file.chunks():
                dest.write(chunk)

        try:
            df = load_dataset(file_path)
            # Get target from request data or detect it
            target = request.data.get('target_column')
            if not target:
                target = detect_target(df)
            else:
                # Use confirmed target
                target = detect_target(df, confirmed_target=target)
            X,y = preprocess_features(df, target)
            best_name, best_model, results = train_models(X, y)

            #Save model
            model_filename = f"model_{uuid.uuid4()}_{best_name.replace(' ', '_')}.pkl"
            model_path = os.path.join(settings.MEDIA_ROOT, model_filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            joblib.dump(best_model, model_path)

            os.remove(file_path)
            os.rmdir(temp_dir)

            #Preparing data for serializer
            response_data = {
                "status": "success",
                "best_model": best_name,
                "results": results,
                "model_download_url": f"/media/{model_filename}",
                "feature_plot_url": f"/plots/feature_importance_{best_name.replace(' ', '_')}.png"
            }

            serializer = TrainingResultSerializer(response_data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Cleanup temp files
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            
            # Return consistent error format
            return Response({
                "status": "error",
                "error": str(e),
                "message": "An error occurred during model training. Please check your CSV file format."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        





    


