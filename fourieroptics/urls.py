from django.urls import path
from . import views

app_name = 'fourieroptics'
urlpatterns = [
    path('', views.home, name='home'),
    
    

]