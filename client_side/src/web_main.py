import dash
import dash_bootstrap_components as dbc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ],
            align="start",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ],
            align="end",
        ),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns"), align="start"),
                dbc.Col(html.Div("One of three columns"), align="center"),
                dbc.Col(html.Div("One of three columns"), align="end"),
            ]
        ),
    ]
)



if __name__ == '__main__':
    app.run_server(debug=True)