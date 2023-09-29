import csv

from django.shortcuts import render, redirect
from django.conf import settings
from .dash_app import PR_app
from .grid_exploration import pareto_grid_app, three_d_app
from .rr_app import RR_app
from .time_app import Time_app
import os
import csv
from django.http import HttpResponse

# Create your views here.

def home(request):
    uploaded_files = request.session.get('uploaded_files', {})
    request.session['uploaded_files'] = uploaded_files
    image_urls = request.session.get('image_urls', [])
    request.session["image_urls"]=image_urls
    if request.method == 'POST':
        file_type = request.POST.get('file_type')
        input_file = request.FILES['input_file']
        file_path = os.path.join(settings.MEDIA_ROOT,file_type, input_file.name)
        file_name = input_file.name
        if file_type in uploaded_files:
            existing_file_name = uploaded_files[file_type]
            existing_file_path = os.path.join(settings.MEDIA_ROOT, file_type, existing_file_name)
            if os.path.exists(existing_file_path):
                os.remove(existing_file_path)
        try:
            if file_type == 'annotations':
                if not file_name.endswith(".csv"):
                    raise ValueError("File extension for annotations must be .csv")
                csv_data = input_file.read().decode('utf-8')
                csv_reader = csv.DictReader(csv_data.splitlines())
                for row in csv_reader:
                    image_url = row.get('url', '')
                    if image_url:
                        image_urls.append(image_url)
                request.session['image_urls'] = image_urls
            elif file_type == 'configuration':
                if not file_name.endswith((".json","txt")):
                    raise ValueError("File extension for configuration must be .json")
            with open(file_path, 'wb') as destination:
                for chunk in input_file.chunks():
                    destination.write(chunk)
        except ValueError as ve:
            return render(request, 'feedback_dashboard/home.html',{'uploaded_files': request.session['uploaded_files'],'image_urls': request.session['image_urls'],'error_msg':str(ve)})
        except Exception as e:
            return render(request, 'feedback_dashboard/home.html',{'uploaded_files': request.session['uploaded_files'],'image_urls': request.session['image_urls'],'error_msg':str(e)})
        uploaded_files[file_type] = file_name
        request.session['uploaded_files'] = uploaded_files
        if 'image_urls' not in request.session:
            request.session['image_urls']=[]
        both_files_uploaded = 'annotations' in uploaded_files and 'configuration' in uploaded_files
        return render(request, 'feedback_dashboard/home.html',{'uploaded_files': request.session['uploaded_files'],'image_urls': request.session['image_urls'],'both_files_uploaded':both_files_uploaded})
    if 'uploaded_files' in request.session:
        both_files_uploaded = 'annotations' in uploaded_files and 'configuration' in uploaded_files
        return render(request, 'feedback_dashboard/home.html', {'uploaded_files': request.session['uploaded_files'], 'image_urls': request.session['image_urls'],'both_files_uploaded':both_files_uploaded})

    return render(request, "feedback_dashboard/home.html")


def reset(request):
    # Clear the uploaded file info from the session
    uploaded_files = request.session.get('uploaded_files', {})
    for file_type, file_name in uploaded_files.items():
        file_path = os.path.join(settings.MEDIA_ROOT, file_type, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
    request.session.clear()
    return redirect('home')

def generateCsv(request):
    print("in download")
    #content = {"result": order_id, "accessToken": access_token}
    return HttpResponse("Success")

def dashboard(request):
    return render(request, "feedback_dashboard/app.html")
