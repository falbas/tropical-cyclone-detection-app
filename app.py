from dash import Dash, html, dcc, Input, Output, callback, ALL, ctx
import dash_leaflet as dl
import datetime


dt = datetime.datetime.now()

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
                    html.H2("Initial Time"),
                    dcc.Dropdown(dates, dates[0], searchable=False, id="initial-time"),
                    html.Div(id="initial-time-output"),
                    html.Div(id="step-output"),
                ],
            ),
            dl.Map(
                [dl.TileLayer(), dl.LayerGroup(id="marker")],
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
    out = []
    try:
        f = open(f"result/{value}.csv", "r")
        for line in f.readlines():
            line = line.strip().split(",")
            if len(out) == 0:
                out.append([[line[1], line[2], line[3]]])
            elif len(out) > 0 and line[1] != out[len(out) - 1][0][0]:
                out.append([[line[1], line[2], line[3]]])
            else:
                out[len(out) - 1].append([line[1], line[2], line[3]])
        f.close()
    except FileNotFoundError:
        return "No result"

    index = 1
    patched_children = []
    for item in out:
        value = ""
        for i in item:
            if len(value) != 0:
                value += "|"
            value += f"{i[0]},{i[1]},{i[2]}"
        patched_children.append(
            html.Button(
                item[0][0],
                id={"type": "btn-step", "index": index, "value": str(value)},
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
    value = ctx.triggered_id.value.split("|")
    return [
        html.H2("Result"),
        html.Div(
            [
                html.Div(f"[{idx+1}] {x.split(',')[1]}, {x.split(',')[2]}")
                for idx, x in enumerate(value)
            ]
        ),
    ]


@callback(
    Output("marker", "children"),
    Input({"type": "btn-step", "index": ALL, "value": ALL}, "n_clicks"),
)
def add_marker(_):
    if ctx.triggered_id == None:
        return ""
    value = ctx.triggered_id.value.split("|")

    marker = []
    for i in value:
        i = i.split(",")
        marker.append(
            dl.DivMarker(
                position=[float(i[1]), float(i[2])],
                iconOptions=dict(
                    html="<img src='./assets/images/tc.png'/>",
                    className="tc-marker",
                    iconSize=[50, 50],
                ),
            )
        )
    return marker


if __name__ == "__main__":
    app.run(debug=True)
