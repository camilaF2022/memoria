from rest_framework import serializers
from .models import GeneratedDataset, ModeloDataset, ModeloTrain

class GeneratedDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedDataset
        fields = [
            'id', 'name', 'description',
            'data_type', 'sample_count',
            'folder_paths', 'created_at',
            'status',  # 👈 AGREGA ESTO
        ]
        read_only_fields = ['id', 'folder_paths', 'created_at', 'status']


class ModeloDatasetSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    model_type_label = serializers.SerializerMethodField()

    class Meta:
        model = ModeloDataset
        fields = [
            'id', 'name', 'dataset', 'dataset_name',
            'model_type', 'model_type_label',  # 👈 Asegúrate de incluir ambos
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'dataset_name']

    def get_model_type_label(self, obj):
        return obj.get_model_type_display()

class ModeloTrainSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source='model.name', read_only=True)
    model_type = serializers.CharField(source='model.model_type', read_only=True)

    class Meta:
        model = ModeloTrain
        fields = [
            'id', 'training_name', 'model', 'model_name', 'model_type',
            'epochs', 'batch_size', 'accuracy', 'loss', 'status',
            'created_at', 'confusion_path', 'model_path',
            'var_smoothing', 'test_split', 'n_clusters', 'combined_data'
        ]



