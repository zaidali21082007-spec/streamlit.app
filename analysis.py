# analysis.py
"""
analysis.py
------------
Core data-analysis layer for the AI Data Analysis Assistant.

Implements:
    STEP 1 - load_dataset(file_path)
    STEP 2 - analyze_dataset(df)
    STEP 3 - answer_question(df, question)
"""

import os
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    STEP 1: Load a CSV dataset from disk and print a diagnostic summary.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file exists but cannot be parsed as a valid CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[load_dataset] The specified data file was not found: '{file_path}'"
        )

    try:
        loaded_dataframe = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as empty_data_error:
        raise ValueError(
            f"[load_dataset] The file '{file_path}' is empty or contains no parsable columns."
        ) from empty_data_error
    except pd.errors.ParserError as parser_error:
        raise ValueError(
            f"[load_dataset] The file '{file_path}' could not be parsed as valid CSV."
        ) from parser_error

    total_row_count = loaded_dataframe.shape[0]
    total_column_count = loaded_dataframe.shape[1]
    column_name_list = list(loaded_dataframe.columns)
    column_data_types = loaded_dataframe.dtypes
    missing_value_counts_per_column = loaded_dataframe.isnull().sum()

    print("=" * 60)
    print(f"DATASET LOADED SUCCESSFULLY: '{file_path}'")
    print("=" * 60)
    print(f"Total Rows       : {total_row_count}")
    print(f"Total Columns    : {total_column_count}")
    print(f"Column Names     : {column_name_list}")
    print("-" * 60)
    print("Data Types Per Column:")
    print(column_data_types)
    print("-" * 60)
    print("Missing Values Per Column:")
    print(missing_value_counts_per_column)
    print("=" * 60)

    return loaded_dataframe


def analyze_dataset(dataframe: pd.DataFrame) -> dict:
    """
    STEP 2: Dynamically analyze numeric and categorical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataset to analyze.

    Returns
    -------
    dict
        Nested dictionary structured as:
        {
            "total_records": int,
            "numeric_column_statistics": {
                column_name: {"average": ..., "max": ..., "min": ...}
            },
            "categorical_column_distributions": {
                column_name: {value: count, ...}
            }
        }

    Raises
    ------
    ValueError
        If the provided dataframe is empty or None.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("[analyze_dataset] Cannot analyze an empty or missing DataFrame.")

    total_record_count = len(dataframe)

    numeric_column_names = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_column_names = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_column_statistics = {}
    for numeric_column in numeric_column_names:
        column_series = dataframe[numeric_column].dropna()
        if column_series.empty:
            numeric_column_statistics[numeric_column] = {
                "average": None,
                "max": None,
                "min": None,
            }
            continue
        numeric_column_statistics[numeric_column] = {
            "average": float(column_series.mean()),
            "max": float(column_series.max()),
            "min": float(column_series.min()),
        }

    categorical_column_distributions = {}
    for categorical_column in categorical_column_names:
        value_distribution = dataframe[categorical_column].value_counts(dropna=True)
        categorical_column_distributions[categorical_column] = value_distribution.to_dict()

    analysis_results = {
        "total_records": total_record_count,
        "numeric_column_statistics": numeric_column_statistics,
        "categorical_column_distributions": categorical_column_distributions,
    }

    return analysis_results


def answer_question(dataframe: pd.DataFrame, question: str) -> str:
    """
    STEP 3: Answer a fixed set of natural-language validation questions
    using programmatic Pandas logic (string matching + filtering).

    Supported questions:
        a) "Which category appears most frequently?"
        b) "Which item has the maximum Sales value?"
        c) "What is the average of the Sales column?"

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataset to query.
    question : str
        A natural-language question string.

    Returns
    -------
    str
        A human-readable answer to the recognized question, or a
        clarification message if the question is not recognized.

    Raises
    ------
    ValueError
        If the dataframe is empty or None.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("[answer_question] Cannot answer questions on an empty or missing DataFrame.")

    normalized_question = question.strip().lower()

    category_column_name = _find_column_by_keyword(dataframe, "category")
    sales_column_name = _find_column_by_keyword(dataframe, "sales")
    item_column_name = _find_column_by_keyword(dataframe, "product") or _find_column_by_keyword(
        dataframe, "item"
    )

    # Question (a): Most frequent category
    if "most frequently" in normalized_question or (
        "category" in normalized_question and "most" in normalized_question
    ):
        if category_column_name is None:
            return "No categorical 'Category' column was found in the dataset."
        most_frequent_category = dataframe[category_column_name].value_counts().idxmax()
        occurrence_count = dataframe[category_column_name].value_counts().max()
        return (
            f"The most frequently occurring category is '{most_frequent_category}' "
            f"with {occurrence_count} occurrences."
        )

    # Question (b): Item with maximum Sales value
    if "maximum sales" in normalized_question or (
        "sales" in normalized_question and "maximum" in normalized_question
    ):
        if sales_column_name is None:
            return "No 'Sales' column was found in the dataset."
        if item_column_name is None:
            return "No 'Product' or 'Item' identifier column was found to report alongside maximum Sales."
        row_with_max_sales = dataframe.loc[dataframe[sales_column_name].idxmax()]
        return (
            f"The item with the maximum Sales value is "
            f"'{row_with_max_sales[item_column_name]}' with Sales = {row_with_max_sales[sales_column_name]}."
        )

    # Question (c): Average of the Sales column
    if "average" in normalized_question and "sales" in normalized_question:
        if sales_column_name is None:
            return "No 'Sales' column was found in the dataset."
        average_sales_value = dataframe[sales_column_name].mean()
        return f"The average value of the Sales column is {average_sales_value:.2f}."

    return (
        "Question not recognized. Please ask one of the supported questions: "
        "'Which category appears most frequently?', "
        "'Which item has the maximum Sales value?', or "
        "'What is the average of the Sales column?'"
    )


def _find_column_by_keyword(dataframe: pd.DataFrame, keyword: str):
    """
    Internal helper: find a column name containing the given keyword
    (case-insensitive), returning None if no match is found.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataset whose columns will be searched.
    keyword : str
        The keyword to search for within column names.

    Returns
    -------
    str or None
        The matching column name, or None if not found.
    """
    for column_name in dataframe.columns:
        if keyword.lower() in column_name.lower():
            return column_name
    return None
