from django.urls import path
from . import views

urlpatterns = [
    path('rag-answer/',views.rag_answer, name='rag_answer'),
]