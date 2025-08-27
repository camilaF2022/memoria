from django.urls import path, include
from .views import get_protected_data
from .views import logout_view
from rest_framework.routers import DefaultRouter
from core.views import GeneratedDatasetViewSet, ModeloDatasetViewSet,LatestYoloImagesView, ModeloTrainViewSet, GetLabelsView,PredictWithNoiseView, CompareModelsView, KMeansClusteringView,GetModelMetricsView, PredictRandomTensorView, list_user_trainings, register, send_reset_code, reset_password, user_dashboard_stats, user_recent_trainings

router = DefaultRouter()
router.register(r'datasets', GeneratedDatasetViewSet, basename='generatedataset')
router.register(r'models', ModeloDatasetViewSet, basename='trainedmodel')
router.register(r'trains', ModeloTrainViewSet, basename='modelotrain')

urlpatterns = [
    path('protected/', get_protected_data),
    path('logout/', logout_view),
    path('register/', register),
    path('password/send_code/', send_reset_code),
    path('password/reset/', reset_password),
    path('', include(router.urls)),
    path('get_labels/', GetLabelsView.as_view(), name='get_labels'),
    path('predict_with_noise/', PredictWithNoiseView.as_view(), name='predict_with_noise'),
    path('predict_random_tensor/', PredictRandomTensorView.as_view(), name='predict_random_tensor'),
    path('compare_models/', CompareModelsView.as_view(), name='compare_models'),
    path('get_model_metrics/', GetModelMetricsView.as_view(), name='get_model_metrics'),
    path('list_user_trainings/', list_user_trainings, name='list_user_trainings'),
    path('dashboard/stats/', user_dashboard_stats, name='dashboard_stats'),
    path('dashboard/recent_trainings/', user_recent_trainings, name='recent_trainings'),
    path('api/kmeans_cluster/', KMeansClusteringView.as_view(), name='kmeans-cluster'),
    path('latest_yolo_image/', LatestYoloImagesView.as_view(), name='latest_yolo_image'),

]   

