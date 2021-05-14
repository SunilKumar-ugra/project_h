from django.shortcuts import render
import os
from django.views.decorators.csrf import csrf_exempt
from FND import fakeNews
from FND import mainPreprocess
from django.http import HttpResponse
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

# disabling csrf (cross site request forgery)
@csrf_exempt
def getResult(request):
    if request.method == 'GET':
        return render(request, 'fakeNewDetect.html')
    # # if post request came
    if request.method == 'POST':
        # getting values from post
        news = request.POST.get('news')
        response = fakeNews.resolve(news=news)
        print(response)
        # return render(request, 'index.html', {"r": response})
        return render(request, 'fakeNewDetect.html', {"success_msg": response})


# @csrf_exempt
# def API(request):
#     # if post request came
#     if request.method == 'POST':
#         # getting values from post
#         news = request.POST.get('news')
#         print("URL = " + news)
#         response = fakeNews.resolve(news)
#         return HttpResponse(response, status=200)

@csrf_exempt
def logout(request):
    # if request.method == 'POST':
    return render(request, 'index.html')
