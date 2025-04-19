import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import glob
import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)

"""
<scene_id>.edges
two columns containing node IDs : target, source
Note: The tutorial models directed edges with source -> target.


<scene_id>.nodes
seven columns with node properties and target values, which should be predicted
node id, current x, current y, previous x, previous y, future x, future y
the previous x and y represents the location of the pedestrian 1 second ago (you can use those values directly or infer the
movement direction and some speed estimate yourself)
the future x and y represents the target value, i.e., the location where the pedestrian will be in 1 second

Note: Some pedestrians do not have a future x and y coordinate, so you need to filter those for prediction. However, you can
still use their current and previous location when predicting the future location of other pedestrians.

"""
nodes_cols = [
        "node id",
        "current x", "current y",
        "previous x", "previous y",
        "future x",   "future y",
]

edges_cols = ["target", "source"]

def build_dataframe(filename, cols):
    # find all files with the related filename
    pattern   = os.path.join("dataset", f"*.{filename}")
    all_paths = glob.glob(pattern)

    # read each one into a DataFrame
    dfs = []
    cols = cols
    for path in all_paths:
        df = pd.read_csv(
            path,
            sep=",",
            header=None,
            names=cols,
            na_values=["_"],
        )
        dfs.append(df)

    # concatenate them into one big DataFrame
    res = pd.concat(dfs, ignore_index=True)

    print(res)
    return res

nodes = build_dataframe("nodes", nodes_cols)
edges = build_dataframe("edges", edges_cols)
