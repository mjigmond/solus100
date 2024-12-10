import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import rasterio as rio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {name}:{levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# RAM manageable range
rwin = (15000, 16500)
cwin = (20000, 20500)


def raster_to_df(raster_path: Path) -> pd.DataFrame:
    """
    Read raster data into a pandas dataframe.
    :param raster_path: path to raster file
    :return: pandas dataframe with X, Y, and property values
    """
    logger.info(f"processing {raster_path}")
    prop = raster_path.name.split("_")[0]
    with rio.open(raster_path) as ds:
        data = ds.read(1)[rwin[0]:rwin[1],cwin[0]:cwin[1]].astype(float)
        data[data == ds.nodata] = np.nan
        xy = [ds.xy(r, c) for r in range(rwin[0], rwin[1]) for c in range(cwin[0], cwin[1])]
    df = pd.DataFrame(data={prop: data.flatten()})
    x, y = zip(*xy)
    df["X"] = x
    df["Y"] = y
    return df


def build_model(df: pd.DataFrame, target_name: str) -> RandomForestRegressor:
    """
    Builds a scikit-learn regression model.
    :param df: pandas dataframe with features and target
    :param target_name: property to be estimated
    :return: RandomForestRegressor model
    """
    df = df.dropna()
    X = df.drop(target_name, axis=1)
    y = df[target_name]

    np.random.seed(9)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    model = RandomForestRegressor()

    model.fit(X_train, y_train)
    logger.info(f"training data score: {model.score(X_train, y_train):.3f}")
    logger.info(f"test data score: {model.score(X_test, y_test):.3f}")
    pickled_model = Path(f"models/{target_name}.pkl")
    pickled_data = Path(f"models/{target_name}_data.pkl")
    pickle.dump(model, open(pickled_model, "wb"))
    pickle.dump(df, open(pickled_data, "wb"))
    return model
