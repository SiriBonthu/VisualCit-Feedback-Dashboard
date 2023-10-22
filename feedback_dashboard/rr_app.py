import pandas as pd
import os
import numpy as np
import plotly.express as px
from dash import html, dcc, Dash, Patch, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from itertools import product, permutations
from django.conf import settings
from django_plotly_dash import DjangoDash
import json

try:
    RR_app = DjangoDash(name='RR_App')
    try:
        input_data_path = os.path.join(settings.MEDIA_ROOT, 'dashboard_input/dashboard_input.csv')
        input_configuration_path = os.path.join(settings.MEDIA_ROOT, 'dashboard_input/configuration.txt')
    except Exception as e:
        input_data_path = os.path.join(settings.MEDIA_ROOT, 'default_input/dashboard_input.csv')
        input_configuration_path = os.path.join(settings.MEDIA_ROOT, 'default_input/configuration.txt')

    components = []
    annotations = {}
    output = {}
    merged = {}
    global_config = []

    input = pd.read_csv(input_data_path)
    f = open(input_configuration_path)
    input_configuration = json.load(f)
    for action in input_configuration['actions']:
        components.append(action["display_name"])
    for comp in components:
        merged[comp]=input[["id",comp,comp+"_execution_time"]]
        merged[comp]['truth']= input[comp+"_truth"].astype(bool)
        merged[comp].dropna(inplace=True)
    global_config = components
    def format_slider_value(value):
        return f'{value:.1f}'


    for comp in components:
        merged[comp] = merged[comp].dropna()

    def generate_configurations(components):
        configs = [''.join(perm) for perm in permutations(components)]
        return configs


    nInputImages = len(input)
    nValidImages = len(merged[components[0]])

    pipeline_components = {str(i + 1): value for i, value in enumerate(components)}
    thresholds = {key: 0.5 for key in components}
    pipeline_configuration = components

    configs = generate_configurations(pipeline_components.keys())


    def getOutputImageCount(order, threslist):
        count = []
        orderlist = [num for num in order]
        pcomp = pipeline_components[orderlist[0]]
        filterd_df = merged[pcomp][merged[pcomp][pcomp] > threslist[0]]
        count.append(len(filterd_df.index))
        for i in range(1, len(orderlist)):
            pcomp = pipeline_components[orderlist[i]]
            print(pcomp)
            temp_df = merged[pcomp][merged[pcomp]['id'].isin(filterd_df['id'])]
            temp_df = temp_df[temp_df[pcomp] > threslist[i]]
            filterd_df = temp_df
            count.append(len(temp_df.index))
        return count


    def getRR(count, n):
        rr = []
        rr = [round(1 - (value / count[i - 1]) if i > 0 else 1 - (value / n), 2) for i, value in enumerate(count)]
        return rr

    def getCumRR(count,n):
      cum_rr=[]
      cum_rr= [round(value/n,2) for value in count]
      rr=[cum_rr[0]]
      for i in range(1, len(cum_rr)):
        rr.append(cum_rr[i] - cum_rr[i-1])
      return rr


    def getCumulativeReducedImages(count, n):
        cc = []
        cc.append(n),
        cc.extend(count)
        absolute_differences = np.abs(np.diff(cc))
        cum_sums = np.cumsum(absolute_differences)
        return cum_sums.tolist()

    def getGlobalConfig():
        return global_config

    RR_app.layout = html.Div([
        html.H3("Component Thresholds"),
        html.Div(
            id='threshold-sliders',
            children=[
                html.Div([
                    html.H5(component_name),
                    dcc.Slider(
                        id={'type': 'slider', 'component': component_id},
                        min=0,
                        max=1,
                        step=0.01,
                        value=thresholds[component_name],
                        marks={str(threshold): format_slider_value(threshold) for threshold in np.arange(0, 1.1, 0.1)},
                    )
                ], style={
                    'width': '50%',
                    'display': 'inline-block'
                })
                for component_id, component_name in pipeline_components.items()
            ]
        ),
        html.Div(id='selected-threshold'),
        html.Div([html.P(f"Input Images: {nInputImages} Valid Images: {nValidImages}")]),
        html.Div([html.H4(" ".join([f'{key}: {value},' for key, value in pipeline_components.items()]),
                          style={'color': 'blue'})]),
        html.Div([
            dcc.Graph(id="line-graph", style={'display': 'inline-block', 'width': '50%'}),
            dcc.Graph(id="bar-graph", style={'display': 'inline-block', 'width': '50%'}),
        ]),
        html.H3("Component Configurations"),
        html.Div(
            [dcc.Dropdown(
                id='config-dropdown',
                options=[{'label': config, 'value': config} for config in configs],
                placeholder='Select a configuration',
                value='123',
                clearable=False
            ),
            ],
            style={
                'display': 'inline-block',
                'width': '40%'
            }
        ),
        html.Div(id='selected-configuration'),
        html.Div([
            dcc.Graph(id="single-graph", style={'display': 'inline-block', 'width': '50%'}),
        ]),
    ])


    @RR_app.callback(
        Output('selected-configuration', 'children'),
        Output('selected-threshold', 'children'),
        Output('single-graph', 'figure'),
        Output('line-graph', 'figure'),
        Output('bar-graph', 'figure'),
        Input('config-dropdown', 'value'),
        [Input({'type': 'slider', 'component': component_id}, 'value')
         for component_id in pipeline_components]
    )
    def display_selected_configuration(selected_config, *threshold_values):
        for component_id, threshold_value in zip(pipeline_components, threshold_values):
            thresholds[pipeline_components[component_id]] = threshold_value

        line_data = {}
        bar_data = {}
        comp_list = ['comp' + str(i + 1) for i in range(len(components))]
        line_fig = go.Figure()
        for config in configs:
            component_names = [pipeline_components[num] for num in config]
            component_list = ', '.join(component_names)
            value_list = [thresholds[key] for key in component_names]
            line_data[config] = getOutputImageCount(config, value_list)
            #bar_data[config] = getRR(line_data[config], nValidImages)
            cum_sum = getCumulativeReducedImages(line_data[config], nValidImages)
            bar_data[config] = getCumRR(cum_sum, nValidImages)
            line_fig.add_trace(go.Scatter(x=comp_list, y=cum_sum, mode='lines+markers', name=config,
                                          customdata=component_names,
                                          hovertemplate="Cumulative Sum: %{y}<br>Comp: %{customdata}<extra></extra>"))
            line_fig.update_layout(
                title='Reduction of images in different configurations',
                xaxis_title='Components',
                yaxis_title='Cummulative Sum of Images Reduced', )

        # create bar graph
        bar_df = pd.DataFrame(bar_data).transpose()
        bar_df.columns = comp_list
        color_scale = px.colors.sequential.Sunset
        bar_graph = go.Figure()
        for config in bar_df.index:
            grouped_bars = []
            for i in range(len(bar_df.columns)):
                comp = bar_df.columns[i]
                legend_name = pipeline_components[config[i]]
                grouped_bars.append(go.Bar(
                    name=legend_name,
                    x=[config],
                    y=[bar_df.at[config, comp]],
                    customdata=[(legend_name, thresholds[legend_name])],
                    marker_color=color_scale[bar_df.columns.get_loc(comp) % len(color_scale)],
                    hovertemplate='Comp: %{customdata[0]}<br>Thres: %{customdata[1]}<br>RR: %{y}<extra></extra>',
                    showlegend=False))
            for bar in grouped_bars:
                bar_graph.add_trace(bar)
        bar_graph.update_layout(
            title='Reduction rate at component level for different configurations',
            xaxis=dict(title='Pipeline Configuration'),
            yaxis=dict(title='Reduction Rate'),
            barmode='stack')  # Set barmode to 'stack' for stacking bars

        global global_config
        component_names = [pipeline_components[num] for num in selected_config]
        global_config = component_names
        component_list = ', '.join(component_names)
        value_list = [thresholds[key] for key in component_names]
        data = pd.DataFrame()
        data['component'] = ["input"] + component_names
        data['output_images'] = [nValidImages] + getOutputImageCount(selected_config, value_list)
        fig = px.line(data, x="component", y="output_images")
        fig.update_layout(title='filtered output images by each component',
                          xaxis_title='component',
                          yaxis_title='output image count')

        return html.P(f"Selected Configuration: {component_list}"), html.P(
            f"Selected thresholds for components: {thresholds}"), fig, line_fig, bar_graph

except Exception as e:
    print("Exception occurred " + str(e))