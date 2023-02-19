from django.urls import path 
from . import views


urlpatterns = [
    path('', views.index, name='AppDigitalCommunication'),
    path('', views.index, name='AppDigitalCommunication'),
    path('source_coding', views.source_coding, name='source_coding'),
    path('channel_coding', views.channel_coding, name='channel_coding'),
    path('modulation', views.modulation, name='modulation'),
    path('channel', views.channel, name='channel'),
    
    
    
    
]
