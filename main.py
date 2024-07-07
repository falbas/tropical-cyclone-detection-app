from herbie import FastHerbie
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime
import cv2
from ultralytics import YOLO
import os
import rasterio


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# [plotting ecmwf ifs data]

# get date now
dt = datetime.datetime.now()
date = dt.strftime("%Y-%m-%d")
# date = "2024-05-23"

# download data from ecmwf ifs using herbie
print("Get NWP data")
FH = FastHerbie(
    DATES=[date],
    model="ifs",
    product="oper",
    fxx=range(0, 147, 3),
)
# search = ":[u|v]:1000:"
search = ":10[u|v]:"
# FH.download(search)
ds = FH.xarray(search)

# get variable from grib
valid_time = ds["valid_time"]
lat = ds["latitude"][280:441]  # get lat 20, -10
lon = ds["longitude"][1040:1361]  # get lon 80, 160

# generate image of wind direction for yolo detection
os.mkdir(f"input/{date}")
os.mkdir(f"input/{date}-tif")
for i in range(0, len(valid_time)):
    print(f"Generate image {i+1}")
    u = ds["u10"][i][280:441, 1040:1361]
    v = ds["v10"][i][280:441, 1040:1361]

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

    # ws = np.sqrt(u**2 + v**2)  # get wind speed

    transform = rasterio.transform.from_bounds(
        west=lon[0],
        south=lat[-1],
        east=lon[-1],
        north=lat[0],
        width=lon.shape[0],
        height=lat.shape[0],
    )

    with rasterio.open(
        f"input/{date}-tif/{name}-u.tif",
        "w",
        driver="GTiff",
        height=lat.shape[0],
        width=lon.shape[0],
        count=1,
        dtype=u.dtype,
        crs="+proj=latlong",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(u, 1)

    with rasterio.open(
        f"input/{date}-tif/{name}-v.tif",
        "w",
        driver="GTiff",
        height=lat.shape[0],
        width=lon.shape[0],
        count=1,
        dtype=v.dtype,
        crs="+proj=latlong",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(v, 1)


# [detect the tropical cyclone]

os.mkdir(f"output/{date}")
f = open(f"result/{date}.csv", "w")
image_list = os.listdir(f"input/{date}")

# detect the tc and generate image with bounding box and save the result to csv
model = YOLO("model/best.pt")
for i in range(0, len(image_list)):
    print(f"Detecting tropical cyclone {i+1}")
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

        # add bounding box to the image
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(
            img, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        # add circle dot to the center of bounding box
        midx = int(x0 + ((x1 - x0) / 2))
        midy = int(y0 + ((y1 - y0) / 2))
        cv2.circle(img, (midx, midy), 1, (0, 0, 255), 2)

        # get center latlon of the detection result
        w = result[0].orig_shape[1]
        h = result[0].orig_shape[0]
        mlat = round(float(20 - (midy * 40 / h)), 5)
        mlon = round(float((midx * 80 / w) + 80), 5)

        # get bounding box latlon
        wlon = round(float((x0 * 80 / w) + 80), 5)
        nlat = round(float(20 - (y0 * 40 / h)), 5)
        elon = round(float((x1 * 80 / w) + 80), 5)
        slat = round(float(20 - (y1 * 40 / h)), 5)

        # get index of grib latlon
        widx = find_nearest(lon.values, wlon)
        nidx = find_nearest(lat.values, nlat)
        eidx = find_nearest(lon.values, elon)
        sidx = find_nearest(lat.values, slat)

        # get max wind speed of the detection result
        u = ds["u10"][i][nidx:sidx, widx:eidx]
        v = ds["v10"][i][nidx:sidx, widx:eidx]
        ws = np.sqrt(u**2 + v**2)
        maxws = round(float(ws.values.max() * 3.6))

        # write the result
        cv2.imwrite(f"output/{date}/{name}.jpg", img)
        f.write(f"{date},{name},{mlat},{mlon},{score},{maxws}\n")
f.close()
