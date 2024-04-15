from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import requests as rq

""" App initialization part """

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY])

"""Getting data to dataframe """
response = rq.get('http://asterank.com/api/kepler?query={}&limit=2000')
df = pd.json_normalize(response.json())

""" DATA PREPARATION PART """
# Create Star size category
bins = [0, 0.8, 1.2, 100]
names = ['small', 'similar', 'bigger']

df['StarSize'] = pd.cut(df['RSTAR'], bins, labels=names)

options = []
for k in names:
    options.append({'label': k, 'value': k})

# Temperature Bins
tp_bins = [0, 200, 400, 500, 5000]
tp_labels = ['low', 'optimal', 'high', 'extreme']
df['temp'] = pd.cut(df['TPLANET'], tp_bins, labels=tp_labels)

# Planet size
rp_bins = [0, 0.5, 2, 4, 100]
rp_labels = ['low', 'optimal', 'high', 'extreme']
df['gravity'] = pd.cut(df['RPLANET'], rp_bins, labels=rp_labels)

# Estimate object status
df['status'] = np.where((df['temp'] == 'optimal') &
                        (df['gravity'] == 'optimal'),
                        'promising', None)

df.loc[:, 'status'] = np.where((df['temp'] == 'optimal') &
                               (df['gravity'].isin(['low', 'high'])),
                               'challenging', df['status'])

df.loc[:, 'status'] = np.where((df['gravity'] == 'optimal') &
                               (df['temp'].isin(['low', 'high'])),
                               'challenging', df['status'])

df['status'] = df.status.fillna('extreme')

# Relative distance (distance to SUN/SUM radii)
df.loc[:, 'relative_dist'] = df['A'] / df['RSTAR']

""" Design Settings """
CHARTS_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family='Century Gothic', size=14),
        legend=dict(orientation='v',
                    title_text='')
    )
)


"""
Drop Down Filter initialization:
    id - filter id
    options - possible dropdown options
    value - default filter values
    multi (true/false) - On/Off multiselect option in dropdown filter
"""
star_size_selector = dcc.Dropdown(
    id='star_size_selector',
    options=options,
    value=['small', 'similar', 'bigger'],
    multi=True
)
"""
Range Slicer Filter initialization:
    id - slicer id
    min - min value in range slicer
    max - max value in range slicer
    marks - key points in range slicer
    step - slicer step
    value - default slicer value
"""

rplanet_selector = dcc.RangeSlider(
    id='planet_range_slider',
    min=min(df['RPLANET']),
    max=max(df['RPLANET']),
    marks={10: '10', 20: '20', 30: '30'},
    step=1,
    value=[min(df['RPLANET']), max(df['RPLANET'])]
)

"""
Fill tabs with content
"""

tab1_content = [dbc.Row([
    dbc.Col(
        [
            html.Div(id='dist_chart')
        ],
        width={'size': 6}, style={'margin-top': 20}, md=6
    ),
    dbc.Col(
        [
            html.Div(id='celestial_chart')
        ], style={'margin-top': 20}, md=6
    ),
], style={'margin-bottom': 40}
),
    dbc.Row([
        dbc.Col(
            html.Div(id='relative_distance_chart'), md=6
        ),
        dbc.Col(
            html.Div(id='mstar_tstar_chart'), md=6
        )
    ])]

tab2_content = [
    dbc.Row(html.Div(id='data_table'), style={'margin-top': 20})
]

table_header = [
    html.Thead(html.Tr([html.Th("Field Name"), html.Th("Description")]))
]

"""
Creating tab3 with a static table
"""

exp = {'KOI': 'Object of Interest number',
       'A': 'Semi-major axis (AU)',
       'RPLANET': 'Planetary radius (Earth radii)',
       'RSTAR': 'Stellar radius (Sol radii)',
       'TSTAR': 'Effective temperature of host star as reported in KIC (k)',
       'KMAG': 'Kepler magnitude (kmag)',
       'TPLANET': 'Equilibrium temperature of planet, per Borucki et al. (k)',
       'T0': 'Time of transit center (BJD-2454900)',
       'UT0': 'Uncertainty in time of transit center (+-jd)',
       'UT0': 'Uncertainty in time of transit center (+-jd)',
       'PER': 'Period (days)',
       'UPER': 'Uncertainty in period (+-days)',
       'DEC': 'Declination (@J200)',
       'RA': 'Right ascension (@J200)',
       'MSTAR': 'Derived stellar mass (msol)'
       }

tbl_rows = []
for i in exp:
    tbl_rows.append(html.Tr([html.Td(i), html.Td(exp[i])]))

table_body = [html.Tbody(tbl_rows)]
table = dbc.Table(table_header + table_body, bordered=True)
text = 'Data are sourced from Kelper API via asterank.com'
tab3_content = [dbc.Row(html.A(text, href='https://www.asterank.com/kepler'), style={'margin-top': 20}),
                dbc.Row(html.Div(children=table), style={'margin-top': 20})
                ]

"""
Application Layout initialization (frontend of app):
    app.layout - init. of application frontend part
    html.param - calls to some HTML components
    dbc.param - bootstrap element initialization
"""

app.layout = html.Div([
    # header
    dbc.Row([
        dbc.Col(
            html.H1('Exoplanet Data Visualization'))],
        style={'margin-bottom': 40, 'margin-left': '70px'}),

    html.Div(
        [
            # body
            # filters
            dbc.Row([
                dbc.Col(
                    [
                        html.Div('Select planet main semi-axis'),
                        html.Div(rplanet_selector)
                    ],
                    width={'size': 2}
                ),
                dbc.Col(
                    [
                        html.Div('Star size'),
                        html.Div(star_size_selector),
                    ],
                    width={'size': 3, 'offset': 0.5}
                ),
                dbc.Col(
                    dbc.Button('Apply', id='submit-val', n_clicks=0, className='mr-2'), align='center'
                )
            ], style={'margin-bottom': 40}
            ),
            # tabs init.
            dbc.Tabs([
                dbc.Tab(tab1_content, label='Charts'),
                dbc.Tab(tab2_content, label='Data'),
                dbc.Tab(tab3_content, label='About'),
            ])
        ],
        style=({
            'margin-left': '80px',
            'margin-right': '80px'
        })
    )
])

"""
callback initialization part:
    .callback - send parameters and receive results
    
    Output(component_id= ..., component_property= ...) - what the callback will return:
        comp_id - id of chart; 
        component_property - the entity we're returning
        
    Input(component_id= ..., component_property= ...) - what we want to transmit:
        comp_id - id of entity that will get input values (in our case submit-button value)
        component_property - in out case this is a n_clicks (an integer that represents that number of times the button has been clicked)
        
    State(component_id= ..., component_property= ...) - getting a component value that does not cause a callback update:
    (Inputs will trigger your callback; State do not. If you need the the current “value” - aka State - of other dash components within your callback, you pass them along via State.)
        
"""
@app.callback(
    [
        Output(component_id='dist_chart', component_property='children'),
        Output(component_id='celestial_chart', component_property='children'),
        Output(component_id='relative_distance_chart', component_property='children'),
        Output(component_id='mstar_tstar_chart', component_property='children'),
        Output(component_id='data_table', component_property='children')
    ],
    [
        Input(component_id='submit-val', component_property='n_clicks')
    ],
    [
        State(component_id='planet_range_slider', component_property='value'),
        State(component_id='star_size_selector', component_property='value'),
    ]
)

# Func. that will update charts
def update_charts(n, radius_range, star_size):
    chart_data = df[(df['RPLANET'] > radius_range[0]) &
                    (df['RPLANET'] < radius_range[1]) &
                    (df['StarSize'].isin(star_size))
                    ]
    # Check that at least one value is selected in the filter
    if len(chart_data) == 0:
        return (html.Div('Please select at least one filter option'),
                html.Div(),
                html.Div(),
                html.Div(),
                html.Div()
                )

    # Star size ~ planet temp chart
    fig1 = px.scatter(chart_data, x='TPLANET', y='A')
    fig1.update_layout(template=CHARTS_TEMPLATE)
    html1 = [html.H4('Planet Temperature ~ Distance from the Star'),
             dcc.Graph(figure=fig1)]

    # Celestial coord chart
    fig2 = px.scatter(chart_data, x='RA', y='DEC', size='RPLANET', color='status')  # CELESTIAL CHART
    fig2.update_layout(template=CHARTS_TEMPLATE)
    html2 = [html.H4('Position on the Celestial Sphere'),
             dcc.Graph(figure=fig2)]

    # Relative dist chart
    fig3 = px.histogram(chart_data, x='relative_dist', color='status', barmode='overlay', marginal='violin')
    fig3.add_vline(x=1, y0=0, annotation_text='Earth', line_dash='dot')
    fig3.update_layout(template=CHARTS_TEMPLATE)
    html3 = [html.H4('Relative Distance (AU/Sol radii)'),
             dcc.Graph(figure=fig3)]

    fig4 = px.scatter(chart_data, x='MSTAR', y='TSTAR', size='RPLANET', color='status')
    fig4.update_layout(template=CHARTS_TEMPLATE)
    html4 = [html.H4('Star Mass ~ Star Temperature'),
             dcc.Graph(figure=fig4)]

    # Table chart
    raw_data = chart_data.drop(['relative_dist', 'StarSize', 'ROW'], axis=1)
    tbl = dash_table.DataTable(data=raw_data.to_dict('records'), columns=[{'name': i, 'id': i}
                                                                          for i in raw_data.columns],
                               style_data={'width': '100px', 'maxWidth': '100px', 'minWidth': '100px'},
                               page_size=40,
                               style_header={'textAlign': 'center'})
    html5 = [tbl]

    return html1, html2, html3, html4, html5


if __name__ == '__main__':
    app.run_server(debug=True, port='8056')