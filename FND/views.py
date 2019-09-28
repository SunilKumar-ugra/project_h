from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import os

# Create your views here.
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from FND import mainPreprocess

@csrf_exempt

def getcsvfile(request):
    if request.method == 'GET':
        return render(request,'index.html')
    if request.method == 'POST':
        result = []
        csv = request.FILES['docs'].name
        print("Views",csv,type(csv))

        with open(os.path.dirname(__file__)+'/results.csv','w') as file:
            file.truncate(0)

        response = mainPreprocess.getcsvPath(csv)

        with open(os.path.dirname(__file__)+'/results.csv','r') as file:
            for line in file:
                line = line.rstrip()
                line = line.split(",")
                result.append(line)

        print(result)
        return render(request,'response.html',{"r":result})

        # image = reresponse = test.quest.POST.get('image')
        # response = retrain.
