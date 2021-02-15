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


def free_text_preproc_from_dict(
    df: pd.DataFrame,
    words_to_count: Optional[List[str]] = None,
    cols_to_count: Optional[List[str]] = None,
) -> pd.Dataframe:
    """Extracts keywwords in the list from the text in a specific dataframe column
    Returns a dataframe with count of keywords
    |    | apple | banana| peer  |  ... | lemon|
    |  0 |     1 |     0 |     0 |  ... |    0 |"""
    vec = CountVectorizer()

    # Tokenize and count the whole df, or the subset of cols specified
    if cols_to_count == None:
        X = vec.fit_transform(df.position_type)
        count_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    else:
        X = vec.fit_transform(df[cols_to_count].position_type)
        count_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    # For each keyword, check if it appears on the `vec.get_feature_names()`
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

    def col_transform(df):
        col1, col2, col3 = st.beta_columns([2, 1, 1])
        t1_options = col1.multiselect(
            "Columns to transform",
            df.columns.to_list(),
            df.columns.to_list()[0],
        )

        transformations = (
            "To number",
            "Count of keywords",
            "To datetime",
            "To timedelta",
            "To a boolean",
        )
        t1_transformation = col2.selectbox(
            "Transformation to apply",
            transformations,
        )
        with col3:
            if t1_transformation == transformations[0]:
                st.selectbox(
                    "Target data type ?", ("int64", "float64", "int32", "float32")
                )
            if t1_transformation == transformations[1]:
                st.text_input("keywords")
            if t1_transformation == transformations[2]:
                st.selectbox(" ", ("int64", "float64", "int32", "float32"))
            if t1_transformation == transformations[3]:
                st.selectbox("  ", ("int64", "float64", "int32", "float32"))
            if t1_transformation == transformations[4]:
                st.selectbox("", (""))

        st.text(f"You selected: {t1_options}")
        if t1_transformation == "To number":
            t1_df = numeric_preproc(df, t1_options)
        if t1_transformation == "Count of keywords":
            t1_df = free_text_preproc_from_dict(df[t1_options])
        if t1_transformation == "To datetime":
            pass  # t1_df  # TO DO
        if t1_transformation == "To timedelta":
            pass  # t1_df  # TO DO
        return t1_df

    # df_n = numeric_preproc(df, df.columns.to_list(), target_dtype="float32")
    # st.write("changed dtypes")
    processed_subdf = []
    res = col_transform(df)
    processed_subdf.append(res)
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
    st.write(
        pd.concat(
            [processed_subdf[0].dtypes.to_frame().T, processed_subdf[0].head(3)]
        )
    )
    if st.button("📥 Download as csv!"):


if __name__ == "__main__":
    main()
