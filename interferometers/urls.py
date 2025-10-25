from django.urls import path
from . import views

app_name = 'interferometers'


urlpatterns = [
    path('michelson/', views.michelson_interferometer, name='michelson_interferometer'),
    path('fabry-perot/', views.fabry_perot_interferometer, name='fabry_perot_interferometer'),
    path('mach-zehnder/', views.mach_zehnder_interferometer, name='mach_zehnder_interferometer'),
]