import pandas as pd
import os
import numpy as np
import plotly.express as px
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_curve, precision_recall_fscore_support
from sklearn.datasets import make_classification
from dash import html, dcc, Dash, Patch, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pandas.io.formats import style
import plotly.graph_objects as go
import requests
from io import StringIO
from scipy.spatial import ConvexHull, distance
from itertools import product
from django.conf import settings
from django_plotly_dash import DjangoDash

input_data_path = os.path.join(settings.MEDIA_ROOT, 'input/input_data.csv')
input_data = open(input_data_path, 'r').read()
folder_path = os.path.join(settings.MEDIA_ROOT, 'webservice_output_new')
components = ["PublicPlace", "Photo", "TwoPersons"]
results = {}
for comp in components:
    results[comp] = pd.read_csv(folder_path + "/" + comp + ".csv")
    results[comp] = results[comp][["id", comp + "_execution_time"]]

confidences_merged = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'input/merged.csv'))
agreed = pd.read_csv(os.path.join(settings.MEDIA_ROOT, 'input/agreed.csv'))
confidences_merged = confidences_merged.rename(
    columns={"YOLOv5ObjectDetector": "TwoPersons", "MemeClassifier": "Photo", "SceneClassifier": "PublicPlace"})
confidences_merged.columns
for comp in components:
    confidences_merged = confidences_merged.merge(results[comp], on="id", how="left")

np.random.seed(0)
values = 20
thresholds_list = np.linspace(0.01, 0.99, num=values)

# for p in thresholds_list: print(p)
output_df = {}

for thresholds in product(thresholds_list, repeat=3):
    df1 = confidences_merged[components[0]].to_numpy(copy=True)
    df1[df1 < thresholds[0]] = 0.
    df2 = confidences_merged[components[1]].to_numpy(copy=True)
    df2[df2 < thresholds[1]] = 0.
    df3 = confidences_merged[components[2]].to_numpy(copy=True)
    df3[df3 < thresholds[2]] = 0.

    df_ = confidences_merged.copy()
    # create a column pred (True or False)by taking the minimum value of df1,df2,df3 and testing if it is greater than 0
    df_['pred'] = np.array([df1, df2, df3]).min(axis=0) > 0.
    df_ = df_.dropna(how='all')
    output_df[thresholds] = df_

prec_rec_f1_dict_weighted_avg = {}
# for every threshold combination compare the predicted values and ground truth
for thresholds, thresholded in output_df.items():
    true_and_thresholded = agreed.merge(thresholded, on='id')

    metrics = precision_recall_fscore_support(true_and_thresholded['truth'], true_and_thresholded['pred'],
                                              zero_division=0, labels=[True, False], average='binary', pos_label=True)

    # reduction rate is the % of passing ones
    metrics += (true_and_thresholded['pred'].sum() / len(true_and_thresholded['pred']),)

    prec_rec_f1_dict_weighted_avg[thresholds] = metrics

source_list = [(el[0], el[1], el[2], el[4], k) for k, el in prec_rec_f1_dict_weighted_avg.items()]
prec, rec, f1, surv, key = zip(*source_list)

# fixed threshold initial configuration
a = np.array((0.9, 0.9, 0.9))

norms = []
for b in key:
    norms.append(np.linalg.norm(a - b))
armin = np.argmin(np.array(norms))

print(key[armin])

points = np.array([[el[0], el[1]] for el in prec_rec_f1_dict_weighted_avg.values()])
f1 = [el[2] for el in prec_rec_f1_dict_weighted_avg.values()]

limit = 0.01

hull = ConvexHull(points)
hull_vertices = points[hull.vertices]

#app = Dash(__name__)
pareto_grid_app = DjangoDash(name='ParetoGrid')

gdata = pd.DataFrame({'prec': prec, 'rec': rec, 'surv': surv, 'thres': key})
gdata = gdata.round(3)


def getNearestConfigThreshold(thres_values):
    a = np.array(thres_values)
    norms = []
    for b in key:
        norms.append(np.linalg.norm(a - b))
    armin = np.argmin(np.array(norms))
    print(key[armin])
    return armin


def format_slider_value(value):
    return f'{value:.1f}'


def distance_to_edge(point, edge_start, edge_end):
    edge_vector = edge_end - edge_start
    point_vector = point - edge_start
    edge_length = np.linalg.norm(edge_vector)
    edge_direction = edge_vector / edge_length
    projection = np.dot(point_vector, edge_direction)

    if projection <= 0:
        return np.linalg.norm(point - edge_start)  # Closest to edge start
    elif projection >= edge_length:
        return np.linalg.norm(point - edge_end)  # Closest to edge end
    else:
        perpendicular_vector = point_vector - projection * edge_direction
        return np.linalg.norm(perpendicular_vector)

pareto_grid_app.layout = html.Div([
    html.H3('Confidence grid search for precision, recall and reduction rate'),
    html.Div(
        children=[
            html.Div([
                html.H4("SurvivalRate"),
                dcc.Slider(
                    id='reduction-rate-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.2,
                    marks={str(rr): format_slider_value(rr) for rr in np.arange(0, 1.1, 0.1)},
                ),
                html.H4("Grid exploration to estimate pareto frontier"),
                dcc.Graph(id="scatter-plot")
            ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                html.H4("Search for a given configuration"),
                html.Div(id="threshold-inputs",
                         children=[html.Div([
                             html.H5(component),
                             dcc.Input(id=f'{component}-input',
                                       type='number',
                                       placeholder=f'Enter threshold 0-1',
                                       value=0.5,
                                       step=0.1,
                                       min=0,
                                       max=1),
                         ]) for component in components]),
                html.Div(id='message', style={'margin-top': '10px', 'color': 'red'}),
            ], style={'display': 'inline-block', 'width': '20%', 'margin-left': '20px'}
            ),
            html.Div([
                html.H4("Points satisfying the survival rate constraint"),
                dcc.Graph(id="graph2")], style={'display': 'inline-block', 'width': '50%'}
            ),
            html.Div([
                html.H4("Distance from the boundary"),
                dcc.Slider(
                    id='distance-slider',
                    min=0,
                    max=0.1,
                    step=0.01,
                    value=0.05,
                    marks={i / 100: str(i / 100) for i in range(0, 11)},
                ),
                dcc.Graph(id="graph3")], style={'width': '50%', 'display': 'inline-block'}
            ),
            html.Div(id="graph-json")
        ])
])


@pareto_grid_app.callback(
    Output("scatter-plot", "figure"),
    Output("graph2", "figure"),
    Output("graph3", "figure"),
    Output('message', 'children'),
    Input('reduction-rate-slider', 'value'),
    Input('distance-slider', 'value'),
    [Input(f'{component}-input', 'value') for component in components])
def plotGraph(rr, dist, *threshold_values):
    rr = round(rr, 2)
    data = gdata.copy()
    data['thres'] = data['thres'].apply(lambda tpl: tuple(round(val, 2) for val in tpl))
    data["rr"] = [">" + str(rr) if value > rr else "<" + str(rr) for value in surv]
    distances = distance.cdist(data[['prec', 'rec']], hull_vertices)
    min_distances = np.min(distances, axis=1)
    data.loc[(min_distances <= dist) & (data['surv'] >= rr), 'rr'] = 'Optimal'
    # data["rr"][data[min_distances <= dist]]="close to boundary"
    fig = px.scatter(data, x="prec", y="rec", color="rr", hover_data=["surv", "thres"],
                     labels={
                         "x": "Precision",
                         "y": "Recall",
                     }, color_discrete_map={">"+str(rr): 'rgb(250, 196, 132)', "<" + str(rr): 'rgb(243, 231, 155)','Optimal':"rgb(206, 102, 147)"})

    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=0.1,
                                            color='slategrey',
                                            )),
                      selector=dict(mode='markers'))
    fig.update_traces(
        hovertemplate='<b>SurvivalRate:</b> %{customdata[0]}<br><b>Threshold:</b>%{customdata[1]}<br><b>Recall:</b> %{y}<br><b>Precision:</b> %{x}')
    fig.update_xaxes(title_text='Precision')
    fig.update_yaxes(title_text='Recall')

    for simplex in hull.simplices:
        if any(points[simplex, 0] < limit) or any(points[simplex, 1] < limit): continue
        fig.add_trace(
            go.Scatter(x=points[simplex, 0], y=points[simplex, 1],
                       marker={'color': 'slategrey',
                               'size': 0,
                               }, showlegend=False)
        )

    armin = getNearestConfigThreshold(threshold_values)
    fig.add_trace(
        go.Scattergl(x=[prec[armin]], y=[rec[armin]],
                     marker={'color': 'blue',  # firebrick
                             'line_color': 'blue',
                             'size': 10,
                             'symbol': 'cross-thin',
                             'line_width': 4,
                             },
                     name='Selected Configuration')
    )

    fig.update_layout(legend_title_text='Survival Rate', legend=dict(
        tracegroupgap=10,
        itemsizing='constant'
    ))

    fig.update_layout(template='plotly_white')

    fig.update_layout(
        font=dict(size=18),
        yaxis=dict(tickfont=dict(size=18))
    )
    pp = data[data["surv"] >= rr]
    pp[['thres1', 'thres2', 'thres3']] = pp['thres'].apply(pd.Series)
    pp.drop('thres', axis=1, inplace=True)
    fig2 = px.parallel_coordinates(pp, color='surv', color_continuous_scale="sunset")
    fig2.update_layout(
        font=dict(size=14),
    )

    # distances = distance.cdist(pp[['prec', 'rec']], hull_vertices)
    # min_distances = np.min(distances, axis=1)
    # pp = pp[min_distances <= dist]
    min_distances = []
    for point in pp[['prec', 'rec']].values:
        # Initialize a list to store distances from edges for this point
        distances_to_edges = []

        # Loop through the edges of the convex hull
        for i in range(len(hull.vertices)):
            start_vertex = hull_vertices[i]
            end_vertex = hull_vertices[(i + 1) % len(hull.vertices)]  # Wrap around for the last edge

            # Calculate the distance from the point to the current edge
            dist_to_edge = distance_to_edge(point, start_vertex, end_vertex)
            distances_to_edges.append(dist_to_edge)

        # Find the minimum distance from all edges for this point
        min_distance_to_edges = min(distances_to_edges)

        # Append the minimum distance to the min_distances array
        min_distances.append(min_distance_to_edges)

    pp = pp[np.array(min_distances) <= dist]
    fig3 = px.parallel_coordinates(pp, color='surv', color_continuous_scale="sunset",
                                   title='points close to the boundary')
    fig3.update_layout(
        font=dict(size=14),
    )
    message = tuple(round(x, 2) for x in key[armin])
    formatted_message= ', '.join(map(str, message))

    return fig, fig2, fig3,html.Div("Nearest Configuration: "+ formatted_message)


#app.run_server(mode="inline")
surf_df=pd.DataFrame({"prec":prec,"rec":rec,"rr":surv,'thres': key})

mins = [np.min([el[0], el[1], el[2]]) for el in prec_rec_f1_dict_weighted_avg.keys()]
maxs = [np.max([el[0], el[1], el[2]]) for el in prec_rec_f1_dict_weighted_avg.keys()]

scatter_plot = px.scatter_3d(x=mins, y=maxs, z=f1, color=surv, opacity=0.5,color_continuous_scale='sunset')

scatter_plot.update_layout(
    title='3D Scatter Plot of MIN Threshold, MAX Threshold, and F1 Score',
    scene = dict(
                    xaxis_title='MIN threshold',
                    yaxis_title='MAX threshold',
                    zaxis_title='F1'),
     margin=dict(r=10, b=10, l=10, t=50),
                  coloraxis_colorbar=dict(
        title='surv'  # Rename the color bar label
    ))
scatter_plot.update_traces(
    hovertemplate='MIN: %{x}<br>MAX: %{y}<br>F1: %{z}'
)
scatter_plot.update_traces(marker_line_width=0, marker_size=2)

p_array = np.linspace(0.1, 0.9, 20)
r_array = np.linspace(0.1, 0.9, 20)
s_array = np.zeros((len(p_array), len(r_array)))
c=[]

def get_closest_survival(df, target_prec, target_rec):
    distances = np.sqrt((df['prec'] - target_prec)**2 + (df['rec'] - target_rec)**2)
    closest_index = distances.idxmin()
    closest_survival = df.loc[closest_index, 'rr']
    confidence = df.loc[closest_index, 'thres']
    confidence = tuple(round(num, 2) for num in confidence)
    return closest_survival,confidence

for i, p in enumerate(p_array):
    row = []
    for j, r in enumerate(r_array):
        s_array[i,j],conf = get_closest_survival(surf_df, p, r)
        row.append(conf)
    c.append(row)

min_grid, max_grid = np.meshgrid(p_array,r_array)

surface_plot = go.Figure(data=[go.Surface(z=s_array,x=r_array,y=p_array,customdata=c,hovertemplate = 'Recall: %{x}<br>Precision: %{y}<br>Survival Rate: %{z}<br>Confidences:%{customdata}<extra></extra>')])

surface_plot.update_layout(
    title='3D Surface Plot of Precision,Recall and Survival Rate',
    scene=dict(
        xaxis_title='recall',
        yaxis_title='precision',
        zaxis_title='survival rate',
        xaxis=dict(nticks=10),
        yaxis=dict(nticks=10),
        aspectratio=dict(x=1, y=1, z=0.7),
    ),
)

surface_plot.update_layout(
    margin=dict(l=10, r=10, t=50, b=10)  # Adjust the values as needed
)

surface_plot.update_traces(colorscale='sunset',colorbar=dict(title='Surv'))
three_d_app = DjangoDash(name='3D-App')

three_d_app.layout = html.Div([
    dcc.Graph(id="scatter_plot",figure=scatter_plot,style={'display': 'inline-block', 'width': '50%'}),
    dcc.Graph(id="surface_plot",figure=surface_plot,style={'display': 'inline-block', 'width': '50%'}),
])