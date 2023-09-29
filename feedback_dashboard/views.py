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
    image_urls = request.session.get('image_urls', [])
    if request.method == 'POST':
        file_type = request.POST.get('file_type')
        csv_file = request.FILES['csv_file']
        file_path = os.path.join(settings.MEDIA_ROOT,file_type, csv_file.name)
        file_name = csv_file.name
        uploaded_files[file_type] = file_name
        request.session['uploaded_files'] = uploaded_files
        if file_type == 'annotations':
            csv_data = csv_file.read().decode('utf-8')
            csv_reader = csv.DictReader(csv_data.splitlines())
            for row in csv_reader:
                image_url = row.get('url', '')
                if image_url:
                    image_urls.append(image_url)
            request.session['image_urls'] = image_urls
        with open(file_path, 'wb') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)
        if 'image_urls' not in request.session:
            request.session['image_urls']=[]
        return render(request, 'feedback_dashboard/home.html',{'uploaded_files': request.session['uploaded_files'],'image_urls': request.session['image_urls']})
    if 'uploaded_files' in request.session:
        return render(request, 'feedback_dashboard/home.html', {'uploaded_files': request.session['uploaded_files'], 'image_urls': request.session['image_urls']})

    return render(request, "feedback_dashboard/home.html")


def reset(request):
    # Clear the uploaded file info from the session
    if 'uploaded_files' in request.session:
        del request.session['uploaded_files']
        del request.session['image_urls']
    return redirect('home')

def dashboard(request):
    return render(request, "feedback_dashboard/app.html")
