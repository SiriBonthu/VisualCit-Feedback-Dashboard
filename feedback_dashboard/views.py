import csv

from django.shortcuts import render, redirect
from django.conf import settings
from .dash_app import PR_app
from .grid_exploration import pareto_grid_app, three_d_app
from .rr_app import RR_app
from .time_app import Time_app
import os
import csv
import json
import pandas as pd
import requests
from django.http import HttpResponse, JsonResponse
from io import StringIO

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
                    image_url = row.get('media_url', '')
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
    return redirect('/')

def generateCsv(request):
    print("in download")
    address = '131.175.120.2:7779'
    component_names = []
    results = {}
    try:
        config_file_path = os.path.join(settings.MEDIA_ROOT,"configuration", request.session['uploaded_files']['configuration'])
        with open(config_file_path, 'r') as file:
            json_text = file.read()
            mapping = json.loads(json_text)
            print(mapping)
        annotation_file_path = os.path.join(settings.MEDIA_ROOT,"annotations", request.session['uploaded_files']['annotations'])
        print(annotation_file_path)
        input_data = open(annotation_file_path, 'r').read()
        for action in mapping['actions']:
            action["confidence"] = 0.0
            component_names.append(action["name"])
            params = {'actions': [
                action
            ],

                'column_name': 'media_url',
                'csv_file': input_data
            }

            print(params["actions"])
            action_name = action["name"]
            r = requests.post(url=f'http://{address}/Action/API/FilterCSV', json=params)
            if r.status_code == 200:
                results[action_name] = pd.read_csv(StringIO(r.text))
                print("OK")
            else:
                print("NOT OK")
                raise ValueError(f"API request failed with status code {r.status_code}")
        output_df = pd.read_csv(annotation_file_path)
        for comp in component_names:
            temp = results[comp][["id", comp, comp + "_execution_time"]]
            output_df = output_df.merge(temp, on="id", how="left")
        output_folder_path = os.path.join(settings.MEDIA_ROOT, "webservice_output")
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        output_df.to_csv(output_folder_path+"/dashboard_input.csv", index=False)
    except json.JSONDecodeError as e:
        return JsonResponse({"error": f"Invalid Json File: {str(e)}"}, status=400)
    except ValueError as e:
        return JsonResponse({"error": {str(e)}}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Invalid File: {str(e)}"}, status=500)

    return JsonResponse({"message": "Successful"}, status=200)

def show_graphs(request):
    if request.method == 'POST':
        input_file = request.FILES['dashboard_input_file']
        file_path = os.path.join(settings.MEDIA_ROOT, "dashboard_input", "dashboard_input.csv")
        try:
            if not input_file.name.endswith(".csv"):
                raise ValueError("File extension for input file must be .csv")
            with open(file_path, 'wb') as destination:
                for chunk in input_file.chunks():
                    destination.write(chunk)
        except Exception as e:
            return render(request, "feedback_dashboard/dashboard.html",{'error_msg':str(e)})
    return render(request, "feedback_dashboard/app.html")

def dashboard(request):
    return render(request, "feedback_dashboard/dashboard.html")
