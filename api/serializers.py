from rest_framework import serializers

class TrainingResultSerializer(serializers.Serializer):
    """
    Serializer for the AutoML training response.
    """
    status = serializers.CharField(default="success")
    best_model = serializers.CharField()
    results = serializers.DictField(child=serializers.FloatField(),
                                     help_text="Dictionary of model names and their performance metrics")
    model_download_url = serializers.CharField(help_text="URL to download the trained model")
    feature_plot_url = serializers.CharField(allow_null=True,
        required=False,
        help_text="URL to feature importance plot (if available)")

