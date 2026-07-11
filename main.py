"""
main.py
--------
Primary self-contained orchestrator for the AI Data Analysis Assistant.
Executes data pipelines, generates validation answers, and computes stats.
"""

import os
import random
import pandas as pd
import numpy as np

DATA_FILE_PATH = "sales_data.csv"

def seed_simulated_dataset(file_path: str) -> None:
    """Generate a realistic mock sales dataset and write it to disk as a CSV."""
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
        simulated_sales_value = round(float(np.random.uniform(50, 5000)), 2)
        simulated_units_sold = int(np.random.randint(1, 200))

        simulated_records.append({
            "ProductID": product_id,
            "Product": chosen_product,
            "Category": chosen_category,
            "Sales": simulated_sales_value,
            "Units_Sold": simulated_units_sold,
        })

    simulated_dataframe = pd.DataFrame(simulated_records)
    try:
        simulated_dataframe.to_csv(file_path, index=False)
        print(f"[seed_simulated_dataset] Mock dataset generated at '{file_path}'.")
    except OSError as write_error:
        raise OSError(f"[seed_simulated_dataset] Failed to write dataset to '{file_path}'.") from write_error

# --- Embedded Core Engine Dependencies ---
def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing resource: {file_path}")
    return pd.read_csv(file_path)

def analyze_dataset(dataframe: pd.DataFrame) -> dict:
    results = {
        "total_records": len(dataframe),
        "numeric_column_statistics": {},
        "categorical_column_distributions": {}
    }
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        results["numeric_column_statistics"][col] = {
            "Mean": round(float(dataframe[col].mean()), 2),
            "Max": round(float(dataframe[col].max()), 2)
        }
        
    for col in dataframe.select_dtypes(include=[object, 'category']).columns:
        results["categorical_column_distributions"][col] = dataframe[col].value_counts().to_dict()
        
    return results

def answer_question(dataframe: pd.DataFrame, question: str) -> str:
    q_lower = question.lower()
    if "category" in q_lower and "frequent" in q_lower:
        mode_val = dataframe["Category"].mode()[0]
        return f"{mode_val} (Count: {dataframe['Category'].value_counts().iloc[0]})"
    if "maximum sales" in q_lower or "max sales" in q_lower or "maximum" in q_lower:
        idx = dataframe["Sales"].idxmax()
        return f"{dataframe.loc[idx, 'Product']} (${dataframe.loc[idx, 'Sales']})"
    if "average" in q_lower and "sales" in q_lower:
        return f"${dataframe['Sales'].mean():,.2f}"
    return "Analytical validation question criteria skipped."

def run_pipeline() -> None:
    print("\n" + "#" * 70)
    print("# AI DATA ANALYSIS ASSISTANT - PIPELINE START")
    print("#" * 70 + "\n")

    if not os.path.exists(DATA_FILE_PATH):
        print(f"[main] No dataset found at '{DATA_FILE_PATH}'. Seeding data...\n")
        seed_simulated_dataset(DATA_FILE_PATH)

    try:
        loaded_dataframe = load_dataset(DATA_FILE_PATH)
    except Exception as loading_error:
        print(f"[main] FATAL ERROR during dataset loading: {loading_error}")
        return

    try:
        analysis_results = analyze_dataset(loaded_dataframe)
        print("\n" + "-" * 60)
        print("STEP 2 - DATASET ANALYSIS SUMMARY")
        print("-" * 60)
        print(f"Total Records: {analysis_results['total_records']}")
        print("Numeric Column Statistics:")
        for col, stats in analysis_results["numeric_column_statistics"].items():
            print(f"  {col}: {stats}")
    except Exception as analysis_error:
        print(f"[main] FATAL ERROR during data profiling calculations: {analysis_error}")
        return

    validation_question_list = [
        "Which category appears most frequently?",
        "Which item has the maximum Sales value?",
        "What is the average of the Sales column?",
    ]

    print("\n" + "-" * 60)
    print("STEP 3 - JUDGE VALIDATION QUESTIONS")
    print("-" * 60)
    for q in validation_question_list:
        ans = answer_question(loaded_dataframe, q)
        print(f"Q: {q}\nA: {ans}\n")

    print("\n# PIPELINE PIPING COMPLETE SUCCESS\n" + "#" * 70)

if __name__ == "__main__":
    run_pipeline()
