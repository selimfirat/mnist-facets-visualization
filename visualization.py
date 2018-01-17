from matplotlib import pyplot as plot
import random
import numpy as np
import pandas as pd
from PIL import Image as pi
from IPython.core.display import HTML, display
classes =[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]
def map_classes(l):
    return classes[l]

def create_atlas(imgs):
    w, h = int(imgs.shape[1] ** 0.5), int(imgs.shape[1] ** 0.5)
    imgs = imgs.reshape([-1, w, h]) * 255
    c = int(imgs.shape[0] ** 0.5)
    tw = w * c
    th = h * c
    atlas = pi.new("RGB", (tw, th), (0, 0, 0))
    [atlas.paste(pi.fromarray(imgs[j + c * i, :, :]), (j * w, i * h)) for j in range(c) for i in range(c)]

    return atlas


def create_facets(imgs, pred, actual, dataset_name, file_name):
    create_atlas(imgs).save("facets/" + dataset_name + "_atlas.jpg", "JPEG")
    pc = {
        'Actual': list(map(map_classes, actual)),
        'Prediction': list(map(map_classes, pred))
    }

    padf = pd.DataFrame(pc)
    jdf = padf.to_json(orient='records')

    HTML_TEMPLATE = """
            <head>
            <link rel="import" href="./facets-jupyter.html"></link>
            </head>
            <facets-dive id="elem" height="800" sprite-image-width="28" sprite-image-height="28" atlas-url="./{dname}_atlas.jpg"></facets-dive>
            <script>
              var data = JSON.parse('{jdf}');
              document.querySelector("#elem").data = data;
            </script>"""
    facets = HTML_TEMPLATE.format(jdf=jdf, dname=dataset_name)
    f = open("facets/" + dataset_name + "_" + file_name + ".html", "w")
    f.write(str(facets))
    f.close()
    return facets
