""" ### Utilities
    A rather bare script, just for labeling new images if you have them.
"""

import os
from skimage import io
import pandas as pd


def label():
    "A simple function for adding new data"
    files = sorted(os.listdir(config.DATA_DIR))
    tot = len(files)
    y = []
    for i, f in enumerate(files): 
        file_path = os.path.join(data_path, f)
        io.imshow(file_path)
        inp = input(f"{i} of {tot}. Number of squares in image: ")
        try:
            n = int(inp)
            y.append(n)
        except Exception:
            raise Exception
    df_out = pd.DataFrame({"filenames": files, "target": y})
    df_out.to_csv("input/new_data.csv", index=False)
