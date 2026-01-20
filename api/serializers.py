from rest_framework import serializers


class PreviewDataSerializer(serializers.Serializer):
    "Data for paginated preview of uploaded dataset"
    status = serializers.CharField(default="success")
    columns = serializers.ListField(child=serializers.CharField())
    total_rows = serializers.IntegerField()
    offset = serializers.IntegerField()
    limit = serializers.IntegerField()
    returned_rows = serializers.IntegerField()
    data = serializers.ListField(child=serializers.DictField())


class UploadFileResponseSerializer(serializers.Serializer):
    """
    Response after successful file upload
    """
    message = serializers.CharField()
    uploaded_filename = serializers.CharField()
    total_rows = serializers.IntegerField(allow_null=True)
    columns = serializers.ListField(child=serializers.CharField(), allow_null=True)
   
    
class AnalyzeTargetResponseSerializer(serializers.Serializer):
    """Response after target detection/analysis"""
    status = serializers.CharField(default="success")
    uploaded_filename = serializers.CharField()
    detected_target = serializers.CharField()
    all_columns = serializers.ListField(child=serializers.CharField())
    candidate_columns = serializers.ListField(child=serializers.CharField())



class TrainingResultSerializer(serializers.Serializer):
    """Final training result"""
    status = serializers.CharField(default="success")
    best_model = serializers.CharField()
    results = serializers.DictField(child=serializers.FloatField())
    model_download_url = serializers.CharField()
    feature_plot_url = serializers.CharField(allow_null=True)


class ErrorResponseSerializer(serializers.Serializer):
    """Standard error format"""
    status = serializers.CharField(default="error")
    error = serializers.CharField()
    message = serializers.CharField()



