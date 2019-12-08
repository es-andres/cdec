import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import shared_vars as V
import os
import utils

data = utils.load_data()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = 'CDEC'

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id='aggregate_data'),
        html.Div( # title
            [
                html.Div(
                    [
                        html.H2(
                            'Cross Document Event Coreference',

                        ),
                        html.H4(
                            'Model Analysis',
                        )
                    ],

                    className='title'
                ),
            ],
            id="header",
            className='row',
        ),
        html.Div(
            [
                html.H6("Dataset 1"),
                dcc.Dropdown(
                    id='dropdown_1',
                    options=[
                        {'label': 'Model doc. cluster; un-balanced test', 'value': '1_md_ubt'},
                        {'label': 'Model doc. cluster; balanced test', 'value': '1_md_bt'},
                        {'label': 'Gold doc. cluster; un-balanced test', 'value': '1_gd_ubt'},
                        {'label': 'Gold doc. cluster; balanced test', 'value': '1_gd_bt'}
                    ],
                ),
                html.Br(),
                dcc.Dropdown(id='f1_col_picker', multi=True),
                html.Br(),
                html.H6("Dataset 2"),
                dcc.Dropdown(
                    id='dropdown_2',
                    options=[
                        {'label': 'Model doc. cluster; un-balanced test', 'value': '2_md_ubt'},
                        {'label': 'Model doc. cluster; balanced test', 'value': '2_md_bt'},
                        {'label': 'Gold doc. cluster; un-balanced test', 'value': '2_gd_ubt'},
                        {'label': 'Gold doc. cluster; balanced test', 'value': '2_gd_bt'}
                    ],
                ),
                html.Br(),
                dcc.Dropdown(id='f2_col_picker', multi=True),
                html.Div(
                [html.Div(id='box', style={'width': '49%','display':'inline-block'}),
                 html.Div(id='graph', style={'width': '49%','display': 'inline-block'})],
                    style={'width':'100%', 'display': 'inline-block'}
                )
            ],
            id='linegraph_div'
        )
    ]
)

# callbacks
for i in range(1, 3):
    @app.callback(
        [Output(component_id='f{0}_col_picker'.format(i), component_property='options'),
         Output(component_id='f{0}_col_picker'.format(i), component_property='placeholder')],
        [Input(component_id='dropdown_{0}'.format(i), component_property='value')]
    )
    def data_col_picker(input_value):
        if input_value is not None:
            id = input_value.split('_')[0]
            input_value = input_value.replace(id+'_', '')
            if input_value in data:
                opts = [{'label': col, 'value': col} for col in data[input_value].columns.sort_values()]
                return opts, 'Select columns'

            else:
                return [{}], 'Data not yet loaded'
        return [{}], 'Please load a dataset'


@app.callback(
    [Output(component_id='box', component_property='children'),
     Output(component_id='graph', component_property='children')],
    [Input(component_id='f{0}_col_picker'.format(i), component_property='value')
     for i in range(1, 3)],
    [State(component_id='dropdown_{0}'.format(i), component_property='value')
     for i in range(1, 3)]
)
def plot_data(cols_1, cols_2, data_key1, data_key2):
    if cols_1 is not None and cols_2 is None:
        data_key1 = data_key1[2:]
        box = go.Figure()
        graph = go.Figure()
        for c in cols_1:
            box.add_trace(go.Box(y=list(data[data_key1][c].values),
                                 name='(1) ' + c,
                                 boxmean='sd')
                          )
            graph.add_trace(go.Scatter(x=data[data_key1].index.tolist(),
                                       y=list(data[data_key1][c].values),
                                       name='(1) ' + c)
                          )

        box.update_layout(
            yaxis_title='Measure',
            xaxis_title='Dataset'
        )
        graph.update_layout(
            yaxis_title='Measure',
            xaxis_title='Experiment'
        )

        box = dcc.Graph(
            id='b',
            figure=box
        )
        graph = dcc.Graph(
            id='g',
            figure=graph
        ),
        return box, graph
    elif cols_1 is None and cols_2 is not None:
        data_key2 = data_key2[2:]
        box = go.Figure()
        graph = go.Figure()
        for c in cols_2:
            box.add_trace(go.Box(y=list(data[data_key2][c].values),
                                 name='(2) ' + c,
                                 boxmean='sd')
                          )
            graph.add_trace(go.Scatter(x=data[data_key2].index.tolist(),
                                       y=list(data[data_key2][c].values),
                                       name='(2) ' + c)
                          )
        box.update_layout(
            yaxis_title='Measure',
            xaxis_title='Dataset'
        )
        graph.update_layout(
            yaxis_title='Measure',
            xaxis_title='Experiment'
        )

        box = dcc.Graph(
            id='b',
            figure=box
        )
        graph = dcc.Graph(
            id='g',
            figure=graph
        ),
        return box, graph
    elif cols_1 is not None and cols_2 is not None:
        box = go.Figure()
        graph = go.Figure()
        for key in [data_key1, data_key2]:
            id = key[0]
            key = key[2:]
            for c in cols_1 if id == '1' else cols_2:
                box.add_trace(go.Box(y=list(data[key][c].values),
                                     name='({0}) {1}'.format(id, c),
                                     boxmean='sd')
                              )
                graph.add_trace(go.Scatter(x=data[key].index.tolist(),
                                           y=list(data[key][c].values),
                                           name='({0}) {1}'.format(id, c),
                                           line=dict(dash='dash') if id == '1' else dict())
                                )
        box.update_layout(
            yaxis_title='Measure',
            xaxis_title='Dataset'
        )
        graph.update_layout(
            yaxis_title='Measure',
            xaxis_title='Experiment'
        )

        box = dcc.Graph(
            id='b',
            figure=box
        )
        graph = dcc.Graph(
            id='g',
            figure=graph
        ),
        return box, graph
    else:
        return '', ''


# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
