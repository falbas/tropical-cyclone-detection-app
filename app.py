from dash import Dash, html, dcc, Input, Output, callback, ALL, ctx
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import assign
import pandas as pd
import datetime


dt = datetime.datetime.now()

dates = []
for i in range(0, 10):
    date = dt.strftime("%Y-%m-%d")
    dates.append(date)
    dt = dt - datetime.timedelta(days=1)


draw_marker = assign(
    """function(feature, latlng){
    const windMarkerIcon = L.divIcon({className: `tc-marker`, html: `<img src='./assets/images/tc.png'/>`, iconSize: [50, 50]})
    const windMarker = L.marker(latlng, {icon: windMarkerIcon})
    const labelMarkerIcon = L.divIcon({className: `label-marker`, html: `<div>Max wind: ${feature.properties.ws}km/h</div>`})
    const labelMarker = L.marker(latlng, {icon: labelMarkerIcon})
    return L.layerGroup([windMarker, labelMarker])
    }"""
)

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
                    html.H2("Initial Time"),
                    dcc.Dropdown(dates, dates[0], searchable=False, id="initial-time"),
                    html.Div(id="initial-time-output"),
                    html.Div(id="step-output"),
                ],
            ),
            dl.Map(
                [
                    dl.TileLayer(),
                    dl.GeoJSON(id="marker", pointToLayer=draw_marker),
                ],
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
    nested_list = []
    try:
        df = pd.read_csv(
            f"result/{value}.csv",
            header=None,
            names=["date", "step", "lat", "lon", "score", "ws"],
        )
        grouped = df.groupby("step")
        nested_list = [
            group[["step", "lat", "lon", "ws"]].values.tolist()
            for key, group in grouped
        ]
    except:
        return html.P("No result")

    index = 1
    patched_children = []
    for item in nested_list:
        res = ""
        for i in item:
            res += "|" if len(res) != 0 else ""
            res += f"{i[0]},{i[1]},{i[2]},{i[3]}"
        patched_children.append(
            html.Button(
                item[0][0],
                id={"type": "btn-step", "index": index, "value": str(res)},
            )
        )
        index += 1
    return [html.H2("Step"), html.Div(patched_children)]


@callback(
    Output("step-output", "children"),
    Input({"type": "btn-step", "index": ALL, "value": ALL}, "n_clicks"),
)
def display_output(_):
    if ctx.triggered_id == None:
        return ""
    value = [x.split(",") for x in ctx.triggered_id.value.split("|")]
    return [
        html.H2("Result"),
        html.Div(
            [
                html.Div(f"[{idx+1}] {x[1]}, {x[2]}, {x[3]}km/h")
                for idx, x in enumerate(value)
            ]
        ),
    ]


@callback(
    Output("marker", "data"),
    Input({"type": "btn-step", "index": ALL, "value": ALL}, "n_clicks"),
)
def add_geo_marker(_):
    if ctx.triggered_id == None:
        return ""
    value = ctx.triggered_id.value.split("|")

    data = []
    for i in value:
        i = i.split(",")
        data.append(dict(step=i[0], lat=i[1], lon=i[2], ws=i[3]))

    marker = dlx.dicts_to_geojson(data)
    return marker


if __name__ == "__main__":
    app.run(debug=True)
