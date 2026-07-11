# main.py
"""
main.py
--------
Primary orchestrator for the AI Data Analysis Assistant.

Executes the full pipeline in sequence:
    STEP 1 - load_dataset()
    STEP 2 - analyze_dataset()
    STEP 3 - answer_question()   [judge validation questions]
    STEP 4 - generate_visualization()
    STEP 5 - generate_ai_explanation()

If no dataset is present at the expected path, a realistic mock CSV is
auto-generated via seed_simulated_dataset() so the pipeline always runs
successfully out-of-the-box.
"""

import os
import random
import pandas as pd
import numpy as np

from analysis import load_dataset, analyze_dataset, answer_question
from visualization import generate_visualization, generate_ai_explanation


DATA_FILE_PATH = "sales_data.csv"
CHART_OUTPUT_PATH = "analysis_chart.png"


def seed_simulated_dataset(file_path: str) -> None:
    """
    Generate a realistic mock sales dataset and write it to disk as a CSV,
    guaranteeing the pipeline can run end-to-end even with no real data
    file present.

    Parameters
    ----------
    file_path : str
        Destination path where the simulated CSV will be written.

    Returns
    -------
    None
    """
    random.seed(42)
    np.random.seed(42)

    product_category_options = ["Electronics", "Furniture", "Groceries", "Apparel", "Sports"]
    product_name_pool = {
        "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "USB-C Hub", "Laptop Stand"],
        "Furniture": ["Office Chair", "Standing Desk", "Bookshelf", "Filing Cabinet"],
        "Groceries": ["Organic Coffee", "Almond Milk", "Granola Bars", "Olive Oil"],
        "Apparel": ["Running Shoes", "Denim Jacket", "Cotton T-Shirt", "Wool Socks"],
        "Sports": ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Cycling Helmet"],
    }

    total_simulated_rows = 60
    simulated_records = []

    for product_id in range(1, total_simulated_rows + 1):
        chosen_category = random.choice(product_category_options)
        chosen_product = random.choice(product_name_pool[chosen_category])
        simulated_sales_value = round(np.random.uniform(50, 5000), 2)
        simulated_units_sold = int(np.random.randint(1, 200))

        simulated_records.append(
            {
                "ProductID": product_id,
                "Product": chosen_product,
                "Category": chosen_category,
                "Sales": simulated_sales_value,
                "Units_Sold": simulated_units_sold,
            }
        )

    simulated_dataframe = pd.DataFrame(simulated_records)

    try:
        simulated_dataframe.to_csv(file_path, index=False)
        print(f"[seed_simulated_dataset] Mock dataset generated at '{file_path}'.")
    except OSError as write_error:
        raise OSError(
            f"[seed_simulated_dataset] Failed to write simulated dataset to '{file_path}'."
        ) from write_error


def run_pipeline() -> None:
    """
    Execute the full five-step data analysis pipeline end-to-end,
    printing progress and results to the console.

    Returns
    -------
    None
    """
    print("\n" + "#" * 70)
    print("# AI DATA ANALYSIS ASSISTANT - PIPELINE START")
    print("#" * 70 + "\n")

    # Guarantee a dataset exists before attempting to load it.
    if not os.path.exists(DATA_FILE_PATH):
        print(f"[main] No dataset found at '{DATA_FILE_PATH}'. Seeding a simulated dataset...\n")
        seed_simulated_dataset(DATA_FILE_PATH)

    # STEP 1: Load dataset
    try:
        loaded_dataframe = load_dataset(DATA_FILE_PATH)
    except (FileNotFoundError, ValueError) as loading_error:
        print(f"[main] FATAL ERROR during dataset loading: {loading_error}")
        return

    # STEP 2: Analyze dataset
    try:
        analysis_results = analyze_dataset(loaded_dataframe)
        print("\n" + "-" * 60)
        print("STEP 2 - DATASET ANALYSIS SUMMARY")
        print("-" * 60)
        print(f"Total Records: {analysis_results['total_records']}")
        print("Numeric Column Statistics:")
        for column_name, stats_dict in analysis_results["numeric_column_statistics"].items():
            print(f"  {column_name}: {stats_dict}")
        print("Categorical Column Distributions:")
        for column_name, distribution_dict in analysis_results["categorical_column_distributions"].items():
            print(f"  {column_name}: {distribution_dict}")
    except ValueError as analysis_error:
        print(f"[main] FATAL ERROR during dataset analysis: {analysis_error}")
        return

    # STEP 3: Answer fixed validation questions
    validation_question_list = [
        "Which category appears most frequently?",
        "Which item has the maximum Sales value?",
        "What is the average of the Sales column?",
    ]

    print("\n" + "-" * 60)
    print("STEP 3 - JUDGE VALIDATION QUESTIONS")
    print("-" * 60)
    for validation_question in validation_question_list:
        try:
            question_answer = answer_question(loaded_dataframe, validation_question)
        except ValueError as question_error:
            question_answer = f"Could not answer due to an error: {question_error}"
        print(f"Q: {validation_question}\nA: {question_answer}\n")

    # STEP 4: Generate visualization
    try:
        saved_chart_path = generate_visualization(loaded_dataframe, CHART_OUTPUT_PATH)
    except (ValueError, OSError) as visualization_error:
        print(f"[main] ERROR during visualization: {visualization_error}")
        saved_chart_path = None

    # STEP 5: Generate AI-powered executive summary
    print("\n" + "-" * 60)
    print("STEP 5 - AI-GENERATED EXECUTIVE SUMMARY")
    print("-" * 60)
    chart_context_description = (
        f"A bar chart titled 'Total Sales by Category' saved at '{saved_chart_path}'."
        if saved_chart_path
        else "No chart was generated due to a prior error."
    )
    ai_executive_summary = generate_ai_explanation(analysis_results, chart_context_description)
    print(ai_executive_summary)

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
    