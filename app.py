# --- 1. Imports ---
from pathlib import Path
from pickle import UnpicklingError

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- 2. Saved model files ---
# Resolve paths from this script so launching from another directory also works.
MODEL_DIRECTORY = Path(__file__).resolve().parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(model_name: str):
    """Reuse the trusted local classifier and training scaler across reruns."""
    classifier = joblib.load(MODEL_DIRECTORY / MODEL_FILES[model_name])
    scaler = joblib.load(MODEL_DIRECTORY / "scaler.joblib")
    return classifier, scaler


# --- 3. Page setup ---
# Configure the page before displaying any Streamlit elements.
st.set_page_config(page_title="Breast Cancer Model Comparison", layout="wide")

st.title("Breast Cancer Model Comparison")
st.write("Upload a test CSV to get started.")

# --- 4. Sidebar model and positive-class selection ---
st.sidebar.header("Model selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a model",
    options=list(MODEL_FILES),
    key="selected_model",
    help="Evaluate a saved classifier using the uploaded test data.",
)

# Keep selection above the upload stop so it works before a CSV is uploaded.
st.sidebar.info(f"Selected model: {selected_model_name}")
positive_class = st.sidebar.selectbox(
    "Positive class",
    options=[0, 1],
    index=1,
    format_func=lambda label: "1 = Malignant" if label == 1 else "0 = Benign",
    help="Precision, Recall, F1, and AUC measure this class. Predictions stay unchanged.",
)

# --- 5. CSV upload ---
uploaded_file = st.file_uploader("Upload test data", type=["csv"])

# Streamlit reruns this script after an upload; wait until a file is selected.
if uploaded_file is None:
    st.info("Choose a CSV file to preview its contents.")
    st.stop()

# --- 6. Read and validate the uploaded data ---
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

# --- 7. Upload summary and preview ---
st.success("CSV uploaded successfully.")
# DataFrame shape gives the number of rows followed by the number of columns.
st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
st.subheader("Data preview")
st.dataframe(data.head(10), use_container_width=True)

# --- 8. Load the selected model and check evaluation inputs ---
try:
    model, scaler = load_model(selected_model_name)
except (OSError, ValueError, EOFError, UnpicklingError, ImportError) as exc:
    st.error(f"Could not load the saved model or scaler: {exc}")
    st.stop()

if "target" not in data.columns:
    st.error("The CSV must include a 'target' column: 0 = Benign, 1 = Malignant.")
    st.stop()

# Reuse the training column names and order; never fit a scaler on uploaded data.
feature_names = list(scaler.feature_names_in_)
missing_features = [name for name in feature_names if name not in data.columns]
if missing_features:
    st.error(f"Missing feature columns: {', '.join(missing_features)}")
    st.stop()

try:
    features = data[feature_names].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(data["target"], errors="raise")
except (ValueError, TypeError) as exc:
    st.error(f"Features and target must contain numeric values: {exc}")
    st.stop()

if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target.to_numpy()).all():
    st.error("Features and target must not contain empty, NaN, or infinite values.")
    st.stop()

if not target.isin([0, 1]).all():
    st.error("The target column must contain only 0 (Benign) or 1 (Malignant).")
    st.stop()

# --- 9. Predict using the original training preprocessing ---
try:
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)

    # SVM uses decision scores; the other saved classifiers provide probabilities.
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(positive_class)
        positive_scores = model.predict_proba(scaled_features)[:, positive_index]
    else:
        positive_scores = model.decision_function(scaled_features)
        # Binary decision scores point toward classes_[1]; reverse for class 0.
        if positive_class == model.classes_[0]:
            positive_scores = -positive_scores
except ValueError as exc:
    st.error(f"Could not evaluate the uploaded data: {exc}")
    st.stop()

# --- 10. Display the six evaluation metrics ---
st.subheader(f"Evaluation metrics: {selected_model_name}")
positive_name = "Malignant" if positive_class == 1 else "Benign"
st.caption(f"Positive class: {positive_class} = {positive_name}. Scores use the uploaded rows.")

# AUC requires both classes, but the other metrics can still be displayed.
auc = None
if target.nunique() == 2:
    auc = roc_auc_score(target == positive_class, positive_scores)
else:
    st.warning("AUC is unavailable because the target contains only one class.")

# Explicitly explain the zero convention rather than hiding undefined metrics.
if not (target == positive_class).any() or not (predictions == positive_class).any():
    st.warning(
        "The positive class is absent from targets or predictions. "
        "Undefined Precision, Recall, or F1 values are displayed as 0."
    )
if len(set(target) | set(predictions)) == 1:
    st.warning("MCC is undefined when targets and predictions share only one class; displayed as 0.")

metrics = {
    "Accuracy": accuracy_score(target, predictions),
    "AUC": auc,
    "Precision": precision_score(target, predictions, pos_label=positive_class, zero_division=0),
    "Recall": recall_score(target, predictions, pos_label=positive_class, zero_division=0),
    "F1": f1_score(target, predictions, pos_label=positive_class, zero_division=0),
    "MCC": matthews_corrcoef(target, predictions),
}
for column, (label, value) in zip(st.columns(6), metrics.items()):
    column.metric(label, "N/A" if value is None else f"{value:.4f}")

# --- 11. Selected-model confusion matrix and classification report ---
selected_model_tab = st.tabs(["Selected Model"])[0]
with selected_model_tab:
    st.subheader(f"Classification details: {selected_model_name}")
    class_labels = [0, 1]
    class_names = ["0 = Benign", "1 = Malignant"]
    st.caption(
        "Both classes are always shown in this order. Changing the positive "
        "class above does not change these per-class results."
    )

    st.subheader("Confusion matrix")
    # Fix the label order so single-class uploads still produce a 2-by-2 matrix.
    matrix = confusion_matrix(target, predictions, labels=class_labels)
    matrix_table = pd.DataFrame(
        matrix,
        index=pd.Index(class_names, name="Actual"),
        columns=pd.Index(class_names, name="Predicted"),
    )

    # Keep explanations tied to malignant cases even when the metric switch changes.
    benign_correct, false_alarms, missed_malignant, malignant_correct = matrix.ravel()
    st.info(
        f"Detected {malignant_correct} of {matrix[1].sum()} malignant cases; "
        f"missed {missed_malignant}; flagged {false_alarms} benign cases as malignant. "
        f"Correct predictions: {benign_correct + malignant_correct} of {matrix.sum()}."
    )

    # Cell descriptions make the counts understandable without knowing TP/TN terms.
    cell_descriptions = [
        ["Correctly benign", "False alarm"],
        ["Missed malignant", "Correctly malignant"],
    ]
    # A standalone figure avoids desktop GUI windows in Streamlit's worker thread.
    figure = Figure(figsize=(7, 4))
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    figure.colorbar(image, ax=axis, label="Number of test rows")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted diagnosis",
        ylabel="Actual diagnosis",
    )
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{cell_descriptions[row][column]}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    st.pyplot(figure)

    st.caption(
        "Rows = actual class; columns = predicted class. Diagonal counts are "
        "correct predictions; off-diagonal counts are errors. Darker blue means more rows, "
        "not better performance."
    )
    with st.expander("How to read the confusion matrix"):
        st.markdown(
            "- **Top left:** benign cases correctly predicted benign (true negatives).\n"
            "- **Top right:** benign cases predicted malignant (false positives / false alarms).\n"
            "- **Bottom left:** malignant cases predicted benign (false negatives / missed cases).\n"
            "- **Bottom right:** malignant cases correctly predicted malignant (true positives).\n\n"
            "These terms treat **1 = Malignant** as positive, regardless of the metric "
            "selector above. The matrix shows counts, not probabilities."
        )
    with st.expander("View the exact matrix counts"):
        st.dataframe(matrix_table)

    st.subheader("Classification report")
    # Per-class metrics describe each class separately, not just the positive one.
    report = classification_report(
        target,
        predictions,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    # Overall accuracy is already displayed above; keep support meaningful here.
    report_table = pd.DataFrame(
        {name: scores for name, scores in report.items() if name != "accuracy"}
    ).T
    st.dataframe(
        report_table.style.format(
            {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}
        )
    )
    st.caption(
        "Support = actual rows per class. Macro avg weights classes equally; "
        "weighted avg weights them by support. Overall accuracy is shown above."
    )
    with st.expander("How to read the classification report"):
        st.markdown(
            "Read each class separately:\n\n"
            "- **Precision:** of the rows predicted as this class, how many were correct?\n"
            "- **Recall:** of the actual rows in this class, how many were detected?\n"
            "- **F1-score:** balances precision and recall using their harmonic mean.\n"
            "- **Support:** the number of actual test rows belonging to this class.\n"
            "- **Macro avg:** averages the two class scores with equal weight.\n"
            "- **Weighted avg:** averages them using each class's support as its weight.\n\n"
            "For example, **0.90 means 90%**. For the malignant row, precision is "
            "`correctly malignant / (correctly malignant + false alarms)`; recall is "
            "`correctly malignant / (correctly malignant + missed malignant)`. "
            "If a denominator is zero, the undefined score is displayed as 0 with a warning."
        )
    if (matrix.sum(axis=0) == 0).any() or (matrix.sum(axis=1) == 0).any():
        st.warning(
            "A class is absent from targets or predictions. Undefined per-class "
            "Precision, Recall, or F1 values in the report are displayed as 0."
        )
    st.caption("Results describe the uploaded test data only. Educational use, not a clinical diagnostic tool.")
