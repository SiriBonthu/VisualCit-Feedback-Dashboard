from django.urls import path,include
from .dash_app import PR_app
from .grid_exploration import pareto_grid_app,three_d_app
from .rr_app import RR_app
from .time_app import Time_app

from . import views

urlpatterns = [
    path("home", views.home, name="home"),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('reset/', views.reset, name='reset'),
    path('generate/', views.generateCsv, name='generate'),
]