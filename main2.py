import logging
from pathlib import Path
import os
import pickle

import pandas as pd


from utils2 import build_model, raster_to_df


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {name}:{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

rasters = Path(os.path.expanduser("~/data/solus100/sandtotal")).glob("*.tif")
test_depths = [0, 5, 15, 30, 60, 81, 100, 150]
models_path = Path("models2")
models_path.mkdir(exist_ok=True)


def main():
    """
    Attempt to estimate percent total sand at any depth within 200cm. All feature data
    covers depths of [0, 5, 15, 30, 60, 100, 150].
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
        for r in rasters:
            dfs.append(raster_to_df(r))
        dfs = pd.concat(dfs, axis=0, ignore_index=True)
        model = build_model(dfs, target_name)
    predict_X = dfs.drop(target_name, axis=1).head(8)
    predict_X["X"] = -535000.0
    predict_X["Y"] = 1753000.0
    predict_X["depth"] = test_depths
    pred = model.predict(predict_X)
    logger.info(f"predicted value of `{target_name}`: {list(zip(test_depths, pred.tolist()))}")

if __name__ == "__main__":
    main()