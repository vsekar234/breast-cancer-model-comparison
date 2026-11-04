# --- 1. Imports ---
import pandas as pd
import streamlit as st

# --- 2. Page setup ---
# Configure the page before displaying any Streamlit elements.
st.set_page_config(page_title="Breast Cancer Model Comparison", layout="wide")

st.title("Breast Cancer Model Comparison")
st.write("Upload a test CSV to get started.")

# --- 3. CSV upload ---
uploaded_file = st.file_uploader("Upload test data", type=["csv"])

# Streamlit reruns this script after an upload; wait until a file is selected.
if uploaded_file is None:
    st.info("Choose a CSV file to preview its contents.")
    st.stop()

# --- 4. Read and validate the uploaded data ---
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

# --- 5. Upload summary and preview ---
st.success("CSV uploaded successfully.")
# DataFrame shape gives the number of rows followed by the number of columns.
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.subheader("Data preview")
st.dataframe(data.head(10), use_container_width=True)
