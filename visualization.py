# visualization.py
"""
visualization.py
-----------------
Visualization and AI-narrative layer for the AI Data Analysis Assistant.

Implements:
    STEP 4 - generate_visualization(df, output_img)
    STEP 5 - generate_ai_explanation(stats, chart_context)
"""

import os
import matplotlib
matplotlib.use("Agg")  # Ensures headless/non-GUI rendering for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_visualization(dataframe: pd.DataFrame, output_img: str = "analysis_chart.png") -> str:
    """
    STEP 4: Render a professional bar chart of total Sales per Category
    and save it to disk.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataset containing 'Category' and 'Sales' columns (or close
        equivalents, resolved via keyword matching).
    output_img : str
        File path where the chart image will be saved.

    Returns
    -------
    str
        The path to the saved chart image.

    Raises
    ------
    ValueError
        If the required category or sales columns cannot be located.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("[generate_visualization] Cannot visualize an empty or missing DataFrame.")

    category_column_name = _resolve_column(dataframe, "category")
    sales_column_name = _resolve_column(dataframe, "sales")

    if category_column_name is None or sales_column_name is None:
        raise ValueError(
            "[generate_visualization] Could not locate both a 'Category' and 'Sales' "
            "column in the provided dataset."
        )

    sales_grouped_by_category = (
        dataframe.groupby(category_column_name)[sales_column_name]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6), dpi=150)

    color_palette = sns.color_palette("viridis", n_colors=len(sales_grouped_by_category))

    bar_plot = sns.barplot(
        data=sales_grouped_by_category,
        x=category_column_name,
        y=sales_column_name,
        hue=category_column_name,
        palette=color_palette,
        legend=False,
        ax=axis,
    )

    axis.set_title("Total Sales by Category", fontsize=16, fontweight="bold", pad=20)
    axis.set_xlabel(category_column_name, fontsize=12, fontweight="semibold", labelpad=12)
    axis.set_ylabel(f"Total {sales_column_name}", fontsize=12, fontweight="semibold", labelpad=12)
    axis.tick_params(axis="x", rotation=30)

    for bar_container in bar_plot.containers:
        bar_plot.bar_label(bar_container, fmt="%.0f", padding=3, fontsize=9)

    plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    figure.tight_layout()

    try:
        figure.savefig(output_img, bbox_inches="tight")
    except OSError as save_error:
        raise OSError(
            f"[generate_visualization] Failed to save chart to '{output_img}'."
        ) from save_error
    finally:
        plt.close(figure)

    print(f"[generate_visualization] Chart saved successfully to '{output_img}'.")
    return output_img


def generate_ai_explanation(stats: dict, chart_context: str) -> str:
    """
    STEP 5: Generate a brief, professional three-sentence executive
    summary of the dataset findings using the Groq LLM API
    (model: llama3-8b-8192).

    Authentication is read from the GROQ_API_KEY environment variable
    (falling back to a GROQ_API_KEY value in a local .env file if
    python-dotenv is available). The key is intentionally NOT hardcoded
    in source code — set it in your shell before running:

        export GROQ_API_KEY="your-key-here"        # macOS/Linux
        setx GROQ_API_KEY "your-key-here"           # Windows

    Parameters
    ----------
    stats : dict
        The nested statistics dictionary produced by analyze_dataset().
    chart_context : str
        A short description of the chart that was generated (e.g. its
        file path or title), used to ground the AI's explanation.

    Returns
    -------
    str
        A three-sentence executive summary, or a graceful fallback
        message if the Groq API is unavailable or misconfigured.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        return (
            "[generate_ai_explanation] Skipped: no GROQ_API_KEY environment variable "
            "was found. Set it with `export GROQ_API_KEY=your_key` and re-run to "
            "receive an AI-generated executive summary."
        )

    try:
        from groq import Groq
    except ImportError as import_error:
        return (
            "[generate_ai_explanation] The 'groq' package is not installed. "
            "Run `pip install groq` and re-run the pipeline."
        )

    try:
        groq_client = Groq(api_key=groq_api_key)

        summary_prompt = (
            "You are a professional data analyst. Based on the following dataset "
            f"statistics:\n{stats}\n\nand the following chart context:\n{chart_context}\n\n"
            "Write a brief, professional executive summary in EXACTLY three sentences "
            "explaining the key findings, trends, and any notable outliers. "
            "Do not use bullet points or headers -- plain prose only."
        )

        chat_completion_response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.4,
            max_tokens=300,
        )

        ai_generated_summary = chat_completion_response.choices[0].message.content.strip()
        return ai_generated_summary

    except Exception as groq_error:  # Broad catch: network, auth, rate-limit, etc.
        return (
            "[generate_ai_explanation] Could not generate AI summary due to an error "
            f"communicating with the Groq API: {groq_error}"
        )


def _resolve_column(dataframe: pd.DataFrame, keyword: str):
    """
    Internal helper: locate a column whose name contains the given
    keyword (case-insensitive).

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