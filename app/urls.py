from django.urls import path , include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'app'
urlpatterns = [
    path('', views.home, name='home'),
   
    
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)