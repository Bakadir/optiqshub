from django.urls import path
from . import views

urlpatterns = [
    path('', views.laser_analysis_view, name='laser_analysis_view'),
]
