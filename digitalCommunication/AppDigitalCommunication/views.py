from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'AppDigitalCommunication/index.html')

# def index(request):
#     return render(request, 'Appcoding.html')DigitalCommunication/source_