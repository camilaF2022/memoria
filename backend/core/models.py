from django.db import models
from django.contrib.auth.models import User
import uuid


class PasswordResetCode(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)

class GeneratedDataset(models.Model):
    class StatusType(models.TextChoices):
        PENDIENTE = "pendiente" , "Pendiente"
        EN_PROGRESO = "en_progreso", "En Progreso"
        COMPLETADO = "completado", "Completado"
        FALLIDO = "fallido", "Fallido"      

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    data_type = models.CharField(max_length=50)
    sample_count = models.IntegerField()
    folder_paths = models.JSONField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=StatusType.choices) 
    created_at = models.DateTimeField(auto_now_add=True)


class ModeloDataset(models.Model):
    class ModelType(models.TextChoices):
        CNN = "CNN"
        LSTM = "LSTM"
        NAIVE_BAYES = "NAIVE_BAYES"
        KMEANS = "KMEANS"
        SVM = "SVM"  
        YOLO = "YOLO"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trained_models')
    name = models.CharField(max_length=255)
    dataset = models.ForeignKey(GeneratedDataset, on_delete=models.CASCADE, related_name='trained_models')
    model_type = models.CharField(max_length=20, choices=ModelType.choices)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class ModeloTrain(models.Model):
    class ModelType(models.TextChoices):
        CNN = "cnn"
        LSTM = "lstm"
        NAIVE_BAYES = "naive_bayes"
        KMEANS = "kmeans"
        SVM = "svm"  
        YOLO = "yolo"

    kernel = models.CharField(max_length=20, null=True, blank=True)  # 👈 por si quieres kernel (rbf, linear, etc.)
    C = models.FloatField(null=True, blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trainings')
    training_name = models.CharField(max_length=255)
    model = models.ForeignKey(ModeloDataset, on_delete=models.CASCADE, related_name='trainings', null=True)
    epochs = models.IntegerField(default=30)
    batch_size = models.IntegerField(default=16)
    var_smoothing = models.FloatField(null=True, blank=True)   # 👈 para naive bayes
    test_split = models.FloatField(null=True, blank=True)      # 👈 para naive bayes
    n_clusters = models.IntegerField(null=True, blank=True)    # 👈 para kmeans
    model_path = models.CharField(max_length=500, blank=True, null=True)
    accuracy = models.FloatField(blank=True, null=True)
    loss = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pendiente')
    confusion_path = models.CharField(max_length=300, null=True, blank=True)
    combined_data = models.JSONField(null=True, blank=True)

    def __str__(self):
        model_type = self.model.model_type if self.model else "?"
        return f"{self.training_name} ({model_type})"

