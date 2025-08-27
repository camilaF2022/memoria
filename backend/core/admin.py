from django.contrib import admin
from .models import GeneratedDataset, ModeloDataset, ModeloTrain
import json
from django.utils.safestring import mark_safe
from django.contrib import admin
from .models import ModeloTrain
@admin.register(GeneratedDataset)
class GeneratedDatasetAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name', 'user', 'data_type', 'sample_count', 'created_at', 'folder_paths'
    )
    search_fields = ('name', 'user__username', 'data_type')
    list_filter = ('data_type', 'created_at', 'user')
    readonly_fields = ('folder_paths', 'created_at')

@admin.register(ModeloDataset)
class ModeloDatasetAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name', 'user', 'model_type', 'dataset', 'created_at', 'updated_at'
    )
    search_fields = ('name', 'user__username', 'model_type', 'dataset__name')
    list_filter = ('model_type', 'created_at', 'updated_at', 'user')
    readonly_fields = ('created_at', 'updated_at')



@admin.register(ModeloTrain)
class ModeloTrainAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'training_name',
        'get_model_name',
        'get_model_type',
        'epochs',
        'batch_size',
        'status',
    )
    list_filter = ('status',)
    readonly_fields = ('formatted_combined_data',)

    @admin.display(ordering='model__name', description='Model Name')
    def get_model_name(self, obj):
        return obj.model.name if obj.model else '-'

    @admin.display(ordering='model__model_type', description='Model Type')
    def get_model_type(self, obj):
        return obj.model.model_type if obj.model else '-'

    def formatted_combined_data(self, obj):
        if not obj.combined_data:
            return "No data"
        try:
            pretty = json.dumps(obj.combined_data, indent=2, ensure_ascii=False)
            return mark_safe(f"<pre>{pretty}</pre>")
        except Exception as e:
            return f"Error: {str(e)}"
    formatted_combined_data.short_description = "Combined Data (JSON)"
