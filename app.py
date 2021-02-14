from __future__ import annotations
from typing import List
from typing import Dict
from typing import Optional
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import streamlit as st  # type: ignore

VALID_DTYPES = {'integer'  : 'int64',
                'float'    : 'float64',
                'text'     : 'str',
                'timestamp': 'datetime',
                'timedelta': 'timedelta',
                'category' : 'category'
                }

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
    if type(cols_to_num) == type([]):
        to_num_df = df[cols_to_num].copy()
        to_num_df = to_num_df.astype(target_dtype, copy=True)
        return to_num_df
    elif type(cols_to_num) == type({}):
        to_num_df = df[cols_to_num].copy()
        to_num_df = to_num_df.astype(cols_to_num, copy=False)
        return to_num_df
    else:
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
    st.set_page_config(
        page_title="Ex-stream-ly Cool App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ########################################################################
    #                               SIDEBAR                                #
    ########################################################################
    with st.sidebar.beta_expander("Upload csv"):
        st.write("Upload here youtr csv file")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
            df = parse(uploaded_file).copy()
    ########################################################################
    #                               BODY                                   #
    ########################################################################

    st.markdown("""# Data type clean up""")
    st.markdown("""### Overview of columns and types""")
    st.write(pd.concat([df.dtypes.to_frame().T, df.head(3)]))

    col1, col2 = st.beta_columns([2, 1])

    col1.markdown("""Numerical columns""")
    numeric_cols = df.select_dtypes("number").columns.to_list()
    col1.text(numeric_cols)
    col1.markdown("""Categorical columns""")
    categorical_cols = df.select_dtypes("category").columns.to_list()
    col1.text(categorical_cols)
    col1.markdown("""Text columns""")
    text_cols = df.select_dtypes("object").columns.to_list()
    col1.text(text_cols)

    col2.markdown("""Boolean columns""")
    text_cols = df.select_dtypes("bool").columns.to_list()
    col2.text(text_cols)
    col2.markdown("""Datetime columns""")
    datetime_cols = df.select_dtypes("datetime").columns.to_list()
    col2.text(datetime_cols)
    col2.markdown("""Timedelta columns""")
    timedelta_cols = df.select_dtypes("timedelta").columns.to_list()
    col2.text(timedelta_cols)

    st.markdown("""### Columns selection""")

    options = st.multiselect(
        "Which columns do you want to transform ?",
        df.columns.to_list(),
        df.columns.to_list()[0],
    )
    st.text(f"You selected: {options}")

    for option in options:
        col1, col2 = st.beta_columns(2)

        col2.selectbox(label, options, index=0, format_func=<class 'str'>, key=None)

    # df_n = numeric_preproc(df, df.columns.to_list(), target_dtype="float32")
    # st.write("changed dtypes")


if __name__ == "__main__":
    main()
