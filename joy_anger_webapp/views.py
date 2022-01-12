from django.shortcuts import render
from joy_anger_webapp import predict

def home(request):    
    return render(request, 'main.html')

def result(request):
    sentence = str(request.GET['sentence'])
    result = predict.predict(sentence)

    return render(request, 'result.html', {'result':result})