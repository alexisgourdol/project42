from __future__ import annotations
from typing import List
from typing import Dict
from typing import Optional
from typing import Union
from itertools import count
from sklearn.feature_extraction.text import CountVectorizer
import base64
import pandas as pd
import streamlit as st  # type: ignoregit


# PARSE INPUT FILE
def parse(csv_file: str) -> pd.DataFrame:
    """Reads csv file, returns a dataframe"""
    return pd.read_csv(csv_file)


# NUMERICAL PREPROC
def numeric_preproc(
    df: pd.DataFrame,
    cols_to_num: Union[Optional[List[str]], Optional[Dict[str, str]]] = None,
    target_dtype: str = "float64",
) -> pd.DataFrame:
    """Converts columns in the parameter array into specified dtype.
    Defaults to `int64` if a list of columns are provided. Accepts a
    dictionnary <str> with column names as keys, <str> target dtype as value"""
    if isinstance(cols_to_num, list):
        to_num_df = df[cols_to_num].copy()
        to_num_df = to_num_df.astype(target_dtype, copy=True)
        return to_num_df
    elif isinstance(cols_to_num, dict):
        to_num_df = df[cols_to_num].copy()
        to_num_df = to_num_df.astype(cols_to_num, copy=False)
        return to_num_df
    else:
        return df


def free_text_preproc(
    df: pd.DataFrame,
    words_to_count: Optional[List[str]] = None,
    cols_to_count: Optional[List[str]] = None,
) -> pd.Dataframe:
    """Extracts keywwords in the list from the text in a specific dataframe column
    Returns a dataframe with count of keywords
    |    | apple | banana| peer  |  ... | lemon|
    |  0 |     1 |     0 |     0 |  ... |    0 |"""
    vec = CountVectorizer()

    # Merge all columns or just the provided subset `cols_to_count` as text
    # Tokenize and count
    if cols_to_count == None:
        df_str = df.astype('str').apply(' '.join, axis=1)
    else:
        df_str = df[cols_to_count].astype('str').apply(' '.join, axis=1)
    X = vec.fit_transform(df_str)
    count_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    # For each keyword, check if it appears on the `vec.get_feature_names()`
    words_to_count = [word.lower() for word in words_to_count]
    words_to_count = [word.strip() for word in words_to_count]
    count_df_kw = pd.DataFrame(index=count_df.index)
    for kw in words_to_count:
        # If so, keep the column
        if kw in count_df.columns:
            count_df_kw = count_df_kw.merge(
                count_df.get(kw), left_index=True, right_index=True
            )
    return count_df_kw


def categorical_preproc(
    df: pd.DataFrame, cols_to_encode: Optional[List[str]] = None
) -> pd.Dataframe:
    pass


def date_time_preproc(
    df: pd.DataFrame, cols_to_convert: Optional[List[str]] = None
) -> pd.Dataframe:
    pass


def main():

    c = count()

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

    st.image('separator-blgr-50.png', use_column_width=True)
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

    st.image('separator-blgr-50.png', use_column_width=True)
    st.markdown("""### Columns selection""")

    def col_transform(df):
        col1, col2, col3 = st.beta_columns([2, 1, 1])

        options = col1.multiselect("Columns to transform",df.columns.to_list(),df.columns.to_list()[0],key=int(next(c)))

        transformations = ("To number", "Count of keywords", "To datetime",
                            "To timedelta","To a boolean")
        transformation = col2.selectbox("Transformation to apply",transformations,key=int(next(c)))

        with col3:
            if transformation == transformations[0]:
                params = st.selectbox("Target data type ?", ("int64", "float64", "int32", "float32"),key=int(next(c)))
            if transformation == transformations[1]:
                kw = st.text_input("Comma separated keywords",key=int(next(c)))
                kw = kw.split()
                kw = [word.strip() for word in kw if word not in ('', ' ')]
                params = [word.replace(',','') for word in kw]
            if transformation == transformations[2]:
                params = st.selectbox(" ", (""),key=int(next(c)))
            if transformation == transformations[3]:
                params = st.selectbox(" ", (""),key=int(next(c)))
            if transformation == transformations[4]:
                params = st.selectbox(" ", (""),key=int(next(c)))

        st.text(f"SUMMARY \n Columns: {options} \n Transformation: {transformation} \n Parameters: {params}")

        if transformation == "To number":
            try:
                df = numeric_preproc(df, options)
            except ValueError as e:
                st.error(f"This data type cannot be converted into a number, please change the column or transformation selection [{e}] ")
        if transformation == "Count of keywords":
            df = free_text_preproc(df, words_to_count=params, cols_to_count=options )
        if transformation == "To datetime":
            pass  # df  # TO DO
        if transformation == "To timedelta":
            pass  # df  # TO DO
        return df

    # df_n = numeric_preproc(df, df.columns.to_list(), target_dtype="float32")
    # st.write("changed dtypes")
    processed_subdf = []
    def get_result(df):
        res = col_transform(df)
        processed_subdf.append(res)

    get_result(df)
    # DuplicateWidgetID: There are multiple identical st.multiselect widgets with the same generated key.
    # if st.button("➕ Save transformation and add another"):
    # col_transform(df)
    # if st.button("✅ Finish the job!"):
    # first_df = processed_subdf[0].copy()
    # for df in processed_subdf[1:]:
    # final_df = pd.concat([first_df, df], axis=1)
    # st.write("Save as csv")
    # st.table(col_transform(df))

    st.markdown("""### Resulting table""")
    st.text(f"Result shapex: {processed_subdf[0].shape}")
    st.write(
        pd.concat(
            [processed_subdf[0].dtypes.to_frame().T, processed_subdf[0].head(3)]
        )
    )

    def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        return href

    st.image('separator-blgr-50.png', use_column_width=True)
    st.markdown(get_table_download_link(processed_subdf[0]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
