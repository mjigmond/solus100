import logging
from pathlib import Path
import os
import pickle

import pandas as pd
import numpy as np

from utils import build_model, raster_to_df


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {name}:{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

rasters = Path(os.path.expanduser("~/data/solus100")).glob("*.tif")
props = ["claytotal", "silttotal", "gypsum", "sandtotal"]
models_path = Path("models")
models_path.mkdir(exist_ok=True)


def main():
    """
    Attempt to estimate percent total sand at a depth of 60cm. All feature data
    were downloaded for the 60cm depth.
    :return:
    """
    target_name = "sandtotal"
    pickled_model = models_path / f"{target_name}.pkl"
    pickled_data = models_path / f"{target_name}_data.pkl"
    if pickled_model.exists():
        logger.info("using existing model")
        model = pickle.load(open(pickled_model, "rb"))
        dfs = pickle.load(open(pickled_data, "rb"))
    else:
        dfs = []
        for i, r in enumerate(rasters):
            prop = r.name.split("_")[0]
            if prop not in props:
                continue
            df = raster_to_df(r)
            dfs.append(df)
        dfs = [dfs[0]] + [d.drop(["X", "Y"], axis=1) for d in dfs[1:]]
        dfs = pd.concat(dfs, axis=1)
        model = build_model(dfs, target_name)
    # build a sample dataframe to predict percent total sand
    # and set two features to NaN (not all estimators support NaNs)
    predict_X = dfs.drop(target_name, axis=1).head(1)
    predict_X["X"] = [-535000.0]
    predict_X["Y"] = [1753000.0]
    predict_X["claytotal"] = [np.nan]
    predict_X["silttotal"] = [np.nan]
    pred = model.predict(predict_X)
    logger.info(f"predicted value of `{target_name}`: {pred[0]}")

if __name__ == "__main__":
    main()