import pandas as pd
import os
import numpy as np
import plotly.express as px
from sklearn.metrics import precision_recall_curve, auc,confusion_matrix,roc_curve,precision_recall_fscore_support
from dash import html,dcc,Dash,Patch,no_update
from dash.dependencies import Input, Output, State
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
    merged[comp]=input[["id",comp]]
    merged[comp]['truth']= input[comp+"_truth"].astype(bool)
    #merged[comp].fillna({comp:0, "truth": False},inplace = True)
    merged[comp].dropna(inplace=True)


def format_slider_value(value):
    return f'{value:.1f}'

PR_app = DjangoDash(name='PR_App')

PR_app.layout = html.Div([
    html.H4("Analysis of the individual component outputs", style={"color": "blue"}),
    html.P("Select component:"),
    html.Div(
        [dcc.Dropdown(
            id='dropdown',
            options=components,
            value='PublicPlace',
            clearable=False,
        ),
        ],
        style={'width': '50%'}
    ),
    html.Div([
        dcc.Graph(id="roc_graph", style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id="pr_graph", style={'display': 'inline-block', 'width': '50%'}),
        dcc.Graph(id="prf1_graph", style={'display': 'inline-block', 'width': '50%'}),
        html.Div([
            html.H5("Confidence-Slider"),
            dcc.Slider(
                id='threshold-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                marks={str(threshold): format_slider_value(threshold) for threshold in np.arange(0, 1.1, 0.1)},
            ),
            dcc.Graph(id="heatmap")], style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'top'})
    ])
])


@PR_app.callback(
    Output("roc_graph", "figure"),
    Output("pr_graph", "figure"),
    Output("prf1_graph", "figure"),
    Output("heatmap", "figure"),
    Input('dropdown', "value"), Input('threshold-slider', 'value'))
def display(comp_name, threshold):
    fpr, tpr, thresholds = roc_curve(merged[comp_name]['truth'], merged[comp_name][comp_name], pos_label=1)
    score = auc(fpr, tpr)
    roc_graph = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'),
        hover_data={'Threshold': thresholds},
        color_discrete_sequence=["rgb(160, 89, 160)"])

    roc_graph.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    precision, recall, thresholds = precision_recall_curve(merged[comp_name]['truth'], merged[comp_name][comp_name],
                                                           pos_label=1)

    pr_graph = px.area(
        x=recall, y=precision,
        title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='Recall', y='Precision'),
        color_discrete_sequence=["rgb(206, 102, 147)"]
    )
    pr_graph.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score = f1_score[:-1]

    min_length = min(len(precision), len(recall), len(thresholds))

    # Truncate the arrays to the minimum length
    precision = precision[:min_length]
    recall = recall[:min_length]
    thresholds = thresholds[:min_length]
    pra_data = pd.DataFrame({"prec": precision, "rec": recall, "f1_score": f1_score, "thres": thresholds})
    prf1_graph = px.line(pra_data, x="thres", y=["prec", "rec", "f1_score"],
                         title='Precision, Recall, F1_score  vs. Threshold')
    prf1_graph.update_layout(xaxis_title='Threshold', yaxis_title='Score')

    # closest_index = find_closest_threshold(threshold)
    comp_output = (merged[comp][comp] >= threshold)
    cm = confusion_matrix(merged[comp]["truth"], comp_output)
    df_cm = pd.DataFrame(cm, index=['Actual Positive', 'Actual Negative'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    heatmap = px.imshow(df_cm,
                        labels=dict(x="Predicted", y="Actual", color="Counts"),
                        color_continuous_scale='sunset',
                        zmin=0,
                        zmax=np.max(cm),
                        title='Confusion Matrix')
    heatmap.update_traces(text=df_cm.values, hovertemplate='Value: %{z}<extra></extra>')
    heatmap.update_layout(hovermode='closest')

    return roc_graph, pr_graph, prf1_graph, heatmap
