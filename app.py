import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background: #F5F7FA;
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: #1E3A8A;
}
.subtitle {
    color: #6B7280;
    font-size: 17px;
}
.block-container {
    padding-top: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CORE ANALYTICAL FUNCTIONS
# -----------------------------
def seed_simulated_dataset() -> pd.DataFrame:
    """Generates a realistic mock sales dataset for out-of-the-box operations."""
    np.random.seed(42)
    categories = ["Electronics", "Furniture", "Groceries", "Apparel", "Sports"]
    pool = {
        "Electronics": ["Wireless Mouse", "Bluetooth Speaker", "USB-C Hub", "Laptop Stand"],
        "Furniture": ["Office Chair", "Standing Desk", "Bookshelf", "Filing Cabinet"],
        "Groceries": ["Organic Coffee", "Almond Milk", "Granola Bars", "Olive Oil"],
        "Apparel": ["Running Shoes", "Denim Jacket", "Cotton T-Shirt", "Wool Socks"],
        "Sports": ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Cycling Helmet"],
    }
    
    records = []
    for product_id in range(1, 61):
        cat = np.random.choice(categories)
        prod = np.random.choice(pool[cat])
        sales = round(float(np.random.uniform(50, 5000)), 2)
        units = int(np.random.randint(1, 200))
        records.append({
            "ProductID": product_id, "Product": prod, "Category": cat, "Sales": sales, "Units_Sold": units
        })
    return pd.DataFrame(records)

def local_query_engine(dataframe: pd.DataFrame, target_query: str) -> str:
    """Answers critical structural and metric queries safely using pandas metadata."""
    q_lower = target_query.lower()
    try:
        if "category" in q_lower and ("frequent" in q_lower or "most" in q_lower):
            if "Category" in dataframe.columns:
                top_cat = dataframe["Category"].mode()[0]
                count = dataframe["Category"].value_counts().iloc[0]
                return f"The category appearing most frequently is **{top_cat}** with {count} entries."
        
        if "maximum sales" in q_lower or "max sales" in q_lower:
            if "Sales" in dataframe.columns and "Product" in dataframe.columns:
                idx = dataframe["Sales"].idxmax()
                return f"**{dataframe.loc[idx, 'Product']}** has the maximum Sales value of **${dataframe.loc[idx, 'Sales']:,}**."
        
        if "average" in q_lower and "sales" in q_lower:
            if "Sales" in dataframe.columns:
                mean_sales = dataframe["Sales"].mean()
                return f"The average value within the Sales column matches **${mean_sales:,.2f}**."
        
        return "Query processed. For specific qualitative assertions, please verify your LLM API Key."
    except Exception as e:
        return f"Failed to compute analytic insight: {str(e)}"

# -----------------------------
# APPLICATION HEADER
# -----------------------------
st.markdown("""
<div class="title">📊 AI Data Analysis Assistant</div>
<div class="subtitle">Upload a CSV dataset, explore automated insights, generate interactive charts, and export reports.</div>
<br>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROL INTERFACE
# -----------------------------
st.sidebar.title("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
chart_type = st.sidebar.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Pie Chart"])

st.sidebar.markdown("---")
api_key = st.sidebar.text_input("LLM API Key (Optional)", type="password")
run_triggered = st.sidebar.button("🚀 Run Comprehensive Analysis")

# Establish Working Dataframe Source
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Target Dataset Loaded Successfully!")
else:
    df = seed_simulated_dataset()
    st.info("💡 Displaying pre-seeded simulated engine data. Upload a custom CSV via the sidebar to override.")

# -----------------------------
# METRICS PIPELINE
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", len(df))
c2.metric("Total Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))

st.divider()

# -----------------------------
# INTERACTIVE DATA PREVIEW
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Dataset Preview")
    st.dataframe(df, use_container_width=True, height=400)

with col2:
    st.subheader("ℹ️ Structural Overview")
    with st.expander("Column Registry", expanded=True):
        st.write(list(df.columns))
    with st.expander("Data Schema"):
        st.dataframe(df.dtypes.astype(str).to_frame(name="Datatype"))
    with st.expander("Null Distribution"):
        st.dataframe(df.isnull().sum().to_frame(name="Missing Counts"))

st.divider()

# -----------------------------
# NATURAL LANGUAGE EXECUTION
# -----------------------------
st.subheader("💬 Metric Query Engine")
q1 = st.text_input("Analysis Query 1", value="Which category appears most frequently?")
q2 = st.text_input("Analysis Query 2", value="Which item has the maximum Sales value?")
q3 = st.text_input("Analysis Query 3", value="What is the average of the Sales column?")

if st.button("Answer Queries"):
    for idx, query in enumerate([q1, q2, q3], 1):
        if query:
            ans = local_query_engine(df, query)
            st.info(f"**Query {idx}:** {query}\n\n**Output:** {ans}")

st.divider()

# -----------------------------
# DYNAMIC VISUALIZATION ENGINE
# -----------------------------
st.subheader("📈 Automated Visualization")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()

if len(numeric_cols) > 0:
    x_axis = st.selectbox("Select X Axis (Categorical Recommended)", categorical_cols if categorical_cols else numeric_cols)
    y_axis = st.selectbox("Select Y Axis (Numeric Recommended)", numeric_cols)
    
    # Generate Plots Reactively
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", template="plotly_white")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} Trend over {x_axis}", template="plotly_white")
        else:
            fig = px.pie(df, names=x_axis, values=y_axis, title=f"{y_axis} Share by {x_axis}", template="plotly_white")
            
        st.plotly_chart(fig, use_container_width=True)
        
        # In-memory buffer conversion for seamless down-stream operations
        img_bytes = fig.to_image(format="png")
        st.download_button("⬇️ Download High-Res Chart Image", data=img_bytes, file_name="generated_chart.png", mime="image/png")
    except Exception as visualization_error:
        st.error(f"Could not render custom interactive visualization: {visualization_error}")
else:
    st.warning("Insufficient numeric column configurations found to isolate coordinates.")

st.divider()

# -----------------------------
# EXECUTIVE ANALYSIS EXPANDER
# -----------------------------
st.subheader("🤖 Automated Executive Summary")
if api_key:
    st.success("LLM Configuration parameters detected. Simulating localized context outputs...")
else:
    st.caption("Using default rule-based summary logic. Provide an API key to enable semantic parsing.")

# Generate statistical insight cards safely
if "Category" in df.columns and "Sales" in df.columns:
    summary_data = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    lead_category = summary_data.index[0]
    
    st.markdown(f"""
    * **Distribution Performance**: Individual segments across **{len(df['Category'].unique())}** clear operational categories were tracked.
    * **Primary Operational Driver**: **{lead_category}** represents the single highest transactional concentration volume.
    * **Data Completeness**: Structural missing profiles sit accurately at **{df.isna().sum().sum()}** anomalies.
    """)
else:
    st.markdown("* Basic structure parsed successfully. Review data summary distributions inside export metrics panels.")

st.divider()

# -----------------------------
# EXPORT UTILITIES
# -----------------------------
c_exp1, c_exp2 = st.columns(2)

with c_exp1:
    # Build text/markdown report dynamically in-memory
    report_stream = io.BytesIO()
    report_text = f"AI Data Analysis Executive Summary\n{'='*35}\n\nRows Detected: {len(df)}\nColumns Detected: {len(df.columns)}\nTotal Missing: {df.isna().sum().sum()}"
    report_stream.write(report_text.encode('utf-8'))
    
    st.download_button(
        "📄 Export Executive Report (.txt)",
        data=report_stream.getvalue(),
        file_name="analysis_report.txt",
        mime="text/plain"
    )

with c_exp2:
    st.download_button(
        "📥 Download Summary Metrics (.csv)",
        data=df.describe().to_csv(),
        file_name="summary_metrics.csv",
        mime="text/csv"
    )
