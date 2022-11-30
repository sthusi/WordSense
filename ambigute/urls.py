from django.urls import path

from .views import AmbiguteView
from .views import aiView

app_name = "ambigute"


urlpatterns = [
    path('ambigute/', AmbiguteView.as_view()),
    path('ambigute/<int:pk>', AmbiguteView.as_view()),
    path('ai/',aiView.as_view()),
   
]
