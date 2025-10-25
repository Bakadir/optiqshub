from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'thinfilms'
urlpatterns = [
    path('use/', views.base, name='base'),
    path('', views.home, name='home'),
    path('data/', views.data, name='data'),
    path('result_calc/', views.result_calc, name='result_calc'),
    path('download/<int:layer_id>/', views.download_file, name='download_file'),
    
    path('update_pages/', views.update_pages, name='update_pages'),
    path('result', views.result, name='result'),

    path('download_excel/<str:book>/<str:page>/', views.download_excel, name='download_excel'),

    path('download_excel_common/', views.download_excel_common, name='download_excel_common'),

    path('download_excel_RTA_p/', views.download_excel_RTA_p, name='download_excel_RTA_p'),
    path('download_excel_RTA_s/', views.download_excel_RTA_s, name='download_excel_RTA_s'),
    path('download_excel_RTA_unpolarized/', views.download_excel_RTA_unpolarized, name='download_excel_RTA_unpolarized'),

    path('download_excel_RTA_angle_p/', views.download_excel_RTA_angle_p, name='download_excel_RTA_angle_p'),
    path('download_excel_RTA_angle_s/', views.download_excel_RTA_angle_s, name='download_excel_RTA_angle_s'),
    path('download_excel_RTA_angle_unpolarized/', views.download_excel_RTA_angle_unpolarized, name='download_excel_RTA_angle_unpolarized'),

    path('download_excel_RTA_thick_p/', views.download_excel_RTA_thick_p, name='download_excel_RTA_thick_p'),
    path('download_excel_RTA_thick_s/', views.download_excel_RTA_thick_s, name='download_excel_RTA_thick_s'),
    path('download_excel_RTA_thick_unpolarized/', views.download_excel_RTA_thick_unpolarized, name='download_excel_RTA_thick_unpolarized'),

    path('layers/', views.manage_layers, name='manage_layers'),

    path('led', views.led, name='led'),


]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)