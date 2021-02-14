from __future__ import annotations
from typing import List
from typing import Dict
from typing import Optional
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import streamlit as st  # type: ignore

# PARSE INPUT FILE
def parse(csv_file: str) -> pd.DataFrame:
    """Reads csv file, returns a dataframe"""
    return pd.read_csv(csv_file)


# NUMERICAL PREPROC
def numeric_preproc(
    df: pd.DataFrame,
    cols_to_num: Union[Optional[List[str]], Optional[Dict[str, str]]] = None,
    target_dtype: str = "int64",
) -> pd.DataFrame:
    """Converts columns in the parameter array into specified dtype.
    Defaults to `int64` if a list of columns are provided
    Accepts a dictionnary <str> with column names as keys, <str> target dtype as value"""
    to_num_df = df[cols_to_num].copy()
    if type(cols_to_num) == "list":
        to_num_df = to_num_df[cols_to_num].astype(target_dtype, copy=False)
        return to_num_df
    if type(cols_to_num) == "dict":
        to_num_df = to_num_df.astype(cols_to_num, copy=False)
        return to_num_df

    return df


def free_text_preproc_from_dict(
    df: pd.DataFrame,
    words_to_keep: Union[Optional[List[str]], Optional[Dict[str, str]]] = None,
) -> pd.Dataframe:
    pass


def categorical_preproc(
    df: pd.DataFrame, cols_to_encode: Optional[List[str]] = None
) -> pd.Dataframe:
    pass


def date_time_preproc(
    df: pd.DataFrame, cols_to_convert: Optional[List[str]] = None
) -> pd.Dataframe:
    pass


def main():
    pass


if __name__ == "__main__":
    main()
