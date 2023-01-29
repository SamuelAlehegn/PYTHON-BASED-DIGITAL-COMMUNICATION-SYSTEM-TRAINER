from django.shortcuts import render
from django.http import JsonResponse
import json


# Create your views here.

 

 
def index(request):
    return render(request, 'AppDigitalCommunication/index.html')

def source_coding(request):
    data = request.POST.get('inpAdd')
    print(data)
    return render(request, 'AppDigitalCommunication/source_coding.html', {'data':data})

def channel_coding(request):
    return render(request, 'AppDigitalCommunication/channel_coding.html')
