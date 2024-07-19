# emissions_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('historical_data/', views.historical_data, name='historical_data'),
    path('predict/', views.make_prediction, name='predict'),
    path('plot/', views.display_plot, name='display_plot'),
]
