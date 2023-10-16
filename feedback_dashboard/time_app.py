import pandas as pd
import os
import numpy as np
import plotly.express as px
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_curve, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.datasets import make_classification
from dash import html, dcc, Dash, Patch, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pandas.io.formats import style
import plotly.graph_objects as go
import requests
from io import StringIO
from itertools import product, permutations
from django.conf import settings
from django_plotly_dash import DjangoDash
import json

input_data_path = os.path.join(settings.MEDIA_ROOT, 'dashboard_input/dashboard_input.csv')
input_configuration_path = os.path.join(settings.MEDIA_ROOT, 'dashboard_input/configuration.txt')
f=open(input_configuration_path)
input_configuration = json.load(f)
components = []
for action in input_configuration['actions']:
    components.append(action["display_name"])
input=pd.read_csv(input_data_path)
annotations = {}
output = {}
merged = {}


for comp in components:
    merged[comp]=input[["id",comp,comp+"_execution_time"]]
    merged[comp].dropna(inplace=True)

def format_slider_value(value):
    return f'{value:.1f}'


for comp in components:
    merged[comp] = merged[comp].dropna()

Time_app = DjangoDash(name='Time_App')

fixed_confidences = [0.1, 0.3,0.5, 0.7, 0.9]

permuted_orders = list(permutations(components))
orders_dict = {i+1: order for i, order in enumerate(permuted_orders)}

# Dictionary to store the sum of execution times for each component at different thresholds
sum_times = {}

for confidence in fixed_confidences:
  sum_times[confidence]=list()
  for key,order in orders_dict.items():
    df_copy = merged.copy()
    sum_time=0
    filtered_df = df_copy[order[0]]
    for comp in order:
      filtered_df = df_copy[comp][df_copy[comp]['id'].isin(filtered_df['id'])]
      sum_time += filtered_df[comp+'_execution_time'].sum()
      filtered_df=filtered_df[filtered_df[comp] >= confidence]
    sum_times[confidence].append(sum_time)

# for conf,times  in sum_times.items():
#     print(f"{conf}: {times}")

configList = [i for i in orders_dict.values()]
rows = []
for key, values in sum_times.items():
    for i, value in enumerate(values):
        rows.append([key, value, configList[i]])

dfc1 = pd.DataFrame(rows, columns=['Confidence', 'Time', 'Configuration'])

scatter_plot1 = px.scatter(dfc1, x='Time', y='Confidence', size='Confidence', hover_data=['Configuration'],
                 title='Execution time at different fixed confidence levels for different pipeline configurations', labels={'Time': 'Time', 'Confidence': 'Confidence'},
                 color='Configuration', color_continuous_scale='sunset')

scatter_plot1.update_traces(hovertemplate='Confidence:</b> %{y}<br><b>Time:</b> %{x}')

confidences = np.round(np.linspace(0.01, 0.99, num=10),2)
sum_times2 = {}
for thresholds in product(confidences, repeat=3):
  sum_times2[thresholds]=list()
  for key,order in orders_dict.items():
    df_copy = merged.copy()
    sum_time=0
    filtered_df = df_copy[order[0]]
    for i in range(len(order)):
      comp=order[i]
      filtered_df = df_copy[comp][df_copy[comp]['id'].isin(filtered_df['id'])]
      sum_time += filtered_df[comp+'_execution_time'].sum()
      filtered_df=filtered_df[filtered_df[comp] >= thresholds[i]]
    sum_times2[thresholds].append(sum_time)

rows = []
for key, values in sum_times2.items():
    for i, value in enumerate(values):
        rows.append([key, value, configList[i],i+1])
times_df = pd.DataFrame(rows, columns=['Confidences', 'Time', 'Configuration','ConfigNumber'])
times_df[['thres1', 'thres2','thres3']] = times_df['Confidences'].apply(pd.Series)

scatter_plot2 = px.scatter(times_df, x='Time', y='ConfigNumber', hover_data=['Confidences'],
                 title='Execution time at different combinations of confidence values for different pipeline configurations',labels={'Time': 'Time', 'ConfigNumber': 'Configuration'},
                 color='Configuration',color_continuous_scale='sunset')
#scatter_plot2.show()

avg_time_per_config = times_df.groupby(['ConfigNumber','Configuration'])['Time'].mean().reset_index()
bar_plot = px.bar(
    avg_time_per_config,
    x='ConfigNumber',
    y='Time',
    color='Configuration',
    title='Average Time Values by Configuration',
    labels={'Time': 'Avg Time', 'ConfigNumber': 'Configuration'},
    #width=1000,
    color_continuous_scale='sunset',
)

normalised_bar_df = pd.DataFrame(columns=['component', 'threshold', 'time'])
for comp in components:
  df_copy = merged[comp].copy()
  df_copy[comp]=df_copy[comp].round(1)
  df_copy=df_copy[[comp,comp+"_execution_time"]]
  df_copy = df_copy.groupby(comp,group_keys="False")[comp+"_execution_time"].apply(lambda x: (x - x.mean()) / x.std()).reset_index()
  df_copy['name']=comp
  df_copy.rename(columns={comp:"threshold",comp+"_execution_time":"time","name":"component"},inplace=True)
  normalised_bar_df=pd.concat([normalised_bar_df, df_copy], ignore_index=True)

print(len(normalised_bar_df))
box_plot = px.box(normalised_bar_df, x='threshold', y='time',points="all", color="component",
             title='Box Plot of Normalized Execution Times for Different Confidence Levels of different components',
             labels={'threshold': 'Confidence Level', 'time': 'Normalized Execution Time'})


Time_app.layout = html.Div([
    dcc.Graph(id="scatter_plot1",figure=scatter_plot1,style={'display': 'inline-block', 'width': '80%'}),
    dcc.Graph(id="scatter_plot2",figure=scatter_plot2,style={'display': 'inline-block', 'width': '80%'}),
    dcc.Graph(id="bar_plot",figure=bar_plot,style={'display': 'inline-block', 'width': '60%'}),
    #dcc.Graph(id="box_plot",figure=box_plot,style={'display': 'inline-block', 'width': '100%'}),
])

