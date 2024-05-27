from dash import Dash, html, dcc, Input, Output, callback, ALL, ctx
import dash_leaflet as dl
import datetime


# dt = datetime.datetime.now()
dt = datetime.datetime(2024, 5, 27)

dates = []
for i in range(0, 10):
    date = dt.strftime("%Y-%m-%d")
    dates.append(date)
    dt = dt - datetime.timedelta(days=1)

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = [
    html.Div(
        style={
            "height": "100vh",
            "margin": "-8px",
            "display": "flex",
            "padding": "2rem",
            "boxSizing": "border-box",
            "gap": "2rem",
        },
        children=[
            html.Div(
                style={"width": "20%"},
                children=[
                    html.H1(
                        "Tropical Cyclone Detection",
                        style={"borderBottom": "1px solid"},
                    ),
                    html.P("Initial Time"),
                    dcc.Dropdown(dates, dates[0], searchable=False, id="initial-time"),
                    html.Div(id="initial-time-output", children=[]),
                    html.Div(id="step-output"),
                ],
            ),
            dl.Map(
                dl.TileLayer(),
                center=[-2.50, 117.50],
                zoom=5,
                style={"height": "100%", "width": "80%"},
                id="map",
            ),
        ],
    ),
]


@callback(Output("initial-time-output", "children"), Input("initial-time", "value"))
def display_step(value):
    # patched_children = Patch()
    patched_children = []
    f = open(f"result/{value}.csv", "r")
    index = 1
    for line in f.readlines():
        patched_children.append(
            html.Button(
                line.split(",")[1],
                id={"type": "btn-step", "index": index, "value": line},
            )
        )
        index += 1
    f.close()
    return patched_children


@callback(
    Output("step-output", "children"),
    Input({"type": "btn-step", "index": ALL, "value": ALL}, "n_clicks"),
)
def display_output(_):
    if ctx.triggered_id == None:
        return ""
    value = ctx.triggered_id.value.split(",")
    return html.Div(f"{value[2]}, {value[3]}")


@callback(
    Output("map", "viewport"),
    Input({"type": "btn-step", "index": ALL, "value": ALL}, "n_clicks"),
)
def fly_to(_):
    if ctx.triggered_id == None:
        return ""
    value = ctx.triggered_id.value.split(",")
    return dict(center=[float(value[2]), float(value[3])], zoom=6, transition="flyTo")


if __name__ == "__main__":
    app.run(debug=True)
