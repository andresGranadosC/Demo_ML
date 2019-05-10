from django.urls import include, path
from django.conf.urls import url
from . import views, predictionModelManager
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    # path('', views.contact, name='index'),
    # path('snippet', views.snippet_detail, name='index'),
    path('', views.welcome, name='welcome'),
    # url(r'^testplot1.png$', views.testplot1),
    # url(r'^testplot2.png$', views.testplot2),
    # path('visualizedata', views.visualizedata, name='visualizedata'),
    path('predictResults', predictionModelManager.predictResults, name='predictResults'),
    path('logrprediction', predictionModelManager.logR_Prediction, name='logrprediction'),
    path('getfeaturescore', predictionModelManager.getFeatureScore, name='getfeaturescore'),
    # url(r'^visualizedata/celery_progress/', include('celery_progress.urls')),  # the endpoint is configurable
    # url(r'^celery_progress/', include('celery_progress.urls')),
    # path('visualizedata/visualizePredictionResults', views.visualizePredictionResults, name='visualizePredictionResults'),
    # url(r'^visualizedata/visualizePredictionResults', views.visualizePredictionResults),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
