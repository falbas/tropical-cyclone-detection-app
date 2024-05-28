from herbie import FastHerbie
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime
import cv2
from ultralytics import YOLO
import os


# plotting ecmwf ifs data

dt = datetime.datetime.now()
date = dt.strftime("%Y-%m-%d")

FH = FastHerbie(
    DATES=[date],
    model="ifs",
    product="oper",
    fxx=range(0, 147, 3),
)

search = ":[u|v]:1000:"
FH.download(search)
ds = FH.xarray(search)

valid_time = ds["valid_time"]
lat = ds["latitude"][280:441]  # get lat 20, -10
lon = ds["longitude"][1040:1361]  # get lon 80, 160

os.mkdir(f"input/{date}")
for i in range(0, len(valid_time)):
    u = ds["u"][i][280:441, 1040:1361]
    v = ds["v"][i][280:441, 1040:1361]
    ws = np.sqrt(u**2 + v**2)

    plt.ioff()  # disable interactive mode
    dpi = 100
    zoom = 3
    fig = plt.figure(figsize=(len(lon) * zoom / dpi, len(lat) * zoom / dpi))

    m = Basemap(
        projection="cyl",
        llcrnrlon=lon[0].values,
        llcrnrlat=lat[-1].values,
        urcrnrlon=lon[-1].values,
        urcrnrlat=lat[0].values,
        resolution="i",
    )
    m.drawcoastlines(color="white")

    skip = 3
    q = m.quiver(
        lon[::skip],
        lat[::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        scale_units="xy",
        scale=5,
        width=0.0015,
    )

    plt.tight_layout(pad=0)

    name = str(valid_time[i].values).split(":")[0]
    fig.savefig(f"input/{date}/{name}.jpg", dpi=dpi)
    plt.close()


# detect the tropical cyclone

os.mkdir(f"output/{date}")
f = open(f"result/{date}.csv", "a")
image_list = os.listdir(f"input/{date}")

model = YOLO("model/best.pt")
for i in range(0, len(image_list)):
    name = image_list[i].split(".")[0]

    img = cv2.imread(f"input/{date}/{name}.jpg")

    result = model(img)
    for detection in result[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = model.names[cls]
        label = f"{object_name} {score}"

        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(
            img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        midx = int(x0 + ((x1 - x0) / 2))
        midy = int(y0 + ((y1 - y0) / 2))
        cv2.circle(img, (midx, midy), 1, (0, 0, 255), 2)

        w = result[0].orig_shape[1]
        h = result[0].orig_shape[0]
        lat = round(float(20 - (midy * 40 / h)), 5)
        lon = round(float((midx * 80 / w) + 80), 5)

        cv2.imwrite(f"output/{date}/{name}.jpg", img)
        f.write(f"{date},{name},{lat},{lon},{x0},{y0},{x1},{y1},{score}\n")
f.close()


# generate colored output

valid_time = ds["valid_time"]
lat = ds["latitude"][280:441]
lon = ds["longitude"][1040:1361]

os.mkdir(f"assets/output/{date}")
f = open(f"result/{date}.csv", "r")
output_idx = set()
for line in f.readlines():
    values = line.split(",")
    for i in range(len(valid_time)):
        if values[1] in str(valid_time[i].values):
            output_idx.add(i)
f.close()

for i in output_idx:
    u = ds["u"][i][280:441, 1040:1361]
    v = ds["v"][i][280:441, 1040:1361]
    ws = np.sqrt(u**2 + v**2)

    plt.ioff()  # disable interactive mode
    dpi = 100
    zoom = 3
    fig = plt.figure(figsize=(len(lon) * zoom / dpi, len(lat) * zoom / dpi))

    m = Basemap(
        projection="cyl",
        llcrnrlon=lon[0].values,
        llcrnrlat=lat[-1].values,
        urcrnrlon=lon[-1].values,
        urcrnrlat=lat[0].values,
        resolution="i",
    )
    m.drawcoastlines(color="black")

    cf = plt.contourf(lon, lat, ws, cmap="jet")

    skip = 5
    q = m.quiver(
        lon[::skip],
        lat[::skip],
        u[::skip, ::skip],
        v[::skip, ::skip],
        scale_units="xy",
        scale=7,
        width=0.0015,
    )

    plt.tight_layout(pad=0)

    name = str(valid_time[i].values).split(":")[0]
    fig.savefig(f"assets/output/{date}/{name}.jpg", dpi=dpi)
    plt.close()

output_image_list = os.listdir(f"output/{date}")

model = YOLO("model/best.pt")
for i in range(0, len(output_image_list)):
    name = output_image_list[i].split(".")[0]

    img = cv2.imread(f"input/{date}/{name}.jpg")
    img2 = cv2.imread(f"assets/output/{date}/{name}.jpg")

    result = model(img)
    for detection in result[0].boxes.data:
        x0, y0 = (int(detection[0]), int(detection[1]))
        x1, y1 = (int(detection[2]), int(detection[3]))
        score = round(float(detection[4]), 2)
        cls = int(detection[5])
        object_name = model.names[cls]
        label = f"{object_name} {score}"

        cv2.rectangle(img2, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(
            img2, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

        midx = int(x0 + ((x1 - x0) / 2))
        midy = int(y0 + ((y1 - y0) / 2))
        cv2.circle(img2, (midx, midy), 1, (0, 0, 255), 2)

        w = result[0].orig_shape[1]
        h = result[0].orig_shape[0]
        lat = round(float(20 - (midy * 40 / h)), 5)
        lon = round(float((midx * 80 / w) + 80), 5)

        cv2.imwrite(f"assets/output/{date}/{name}.jpg", img2)
