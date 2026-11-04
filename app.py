# --- 1. Imports ---
import pandas as pd
import streamlit as st

# --- 2. Page setup ---
# Configure the page before displaying any Streamlit elements.
st.set_page_config(page_title="Breast Cancer Model Comparison", layout="wide")

st.title("Breast Cancer Model Comparison")
st.write("Upload a test CSV to get started.")

# --- 3. Sidebar model selection ---
# Match the five classifiers trained and saved by the evaluation notebook.
model_names = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]

st.sidebar.header("Model selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    options=model_names,
    key="selected_model",
    help="Select a classifier. Predictions and evaluation will be added next.",
)

# Keep selection above the upload stop so it works before a CSV is uploaded.
st.sidebar.info(f"Selected model: {selected_model_name}")
st.sidebar.caption("This step selects a model only; it does not run predictions yet.")

# --- 4. CSV upload ---
uploaded_file = st.file_uploader("Upload test data", type=["csv"])

# Streamlit reruns this script after an upload; wait until a file is selected.
if uploaded_file is None:
    st.info("Choose a CSV file to preview its contents.")
    st.stop()

# --- 5. Read and validate the uploaded data ---
try:
    # Read the uploaded file directly into a pandas table (DataFrame).
    data = pd.read_csv(uploaded_file)
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    # Stop here so unreadable files cannot reach the preview section.
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

# A CSV with column headers but no data rows cannot be used.
if data.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

# --- 6. Upload summary and preview ---
st.success("CSV uploaded successfully.")
# DataFrame shape gives the number of rows followed by the number of columns.
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.subheader("Data preview")
st.dataframe(data.head(10), use_container_width=True)
