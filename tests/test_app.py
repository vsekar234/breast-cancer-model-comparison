"""Exercise the Streamlit upload, model selection, and evaluation flow."""

import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


class EvaluationAppTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(ROOT / "test_data.csv")
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)

    def upload(self, data):
        # AppTest has no file-uploader API; supply a fresh in-memory CSV per rerun.
        with patch(
            "streamlit.file_uploader",
            side_effect=lambda *args, **kwargs: BytesIO(data.to_csv(index=False).encode()),
        ), patch("streamlit.pyplot", wraps=st.pyplot) as chart:
            self.app.run()
        self.assertFalse(self.app.exception)
        self.heatmap = chart.call_args.args[0] if chart.called else None

    def test_sidebar_works_without_upload(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.sidebar.selectbox[0].options, list(MODEL_FILES))
        self.assertEqual(self.app.sidebar.selectbox[1].options, ["0 = Benign", "1 = Malignant"])
        self.assertEqual(self.app.sidebar.selectbox[1].value, 1)
        self.assertEqual(len(self.app.metric), 0)

    def test_all_models_and_positive_classes(self):
        self.upload(self.data)
        scaler = joblib.load(ROOT / "model" / "scaler.joblib")
        features = scaler.transform(self.data[list(scaler.feature_names_in_)])
        target = self.data["target"]

        for name, filename in MODEL_FILES.items():
            model = joblib.load(ROOT / "model" / filename)
            predicted = model.predict(features)
            for positive in (0, 1):
                with self.subTest(model=name, positive=positive):
                    self.app.sidebar.selectbox[0].select(name)
                    self.app.sidebar.selectbox[1].select(positive)
                    self.upload(self.data)
                    self.assertFalse(self.app.error)
                    positive_name = "Malignant" if positive == 1 else "Benign"
                    self.assertTrue(
                        any(
                            f"Positive class: {positive} = {positive_name}." in caption.value
                            for caption in self.app.caption
                        )
                    )
                    if hasattr(model, "predict_proba"):
                        scores = model.predict_proba(features)[:, positive]
                    else:
                        scores = model.decision_function(features) * (1 if positive == 1 else -1)
                    expected = {
                        "Accuracy": accuracy_score(target, predicted),
                        "AUC": roc_auc_score(target == positive, scores),
                        "Precision": precision_score(target, predicted, pos_label=positive),
                        "Recall": recall_score(target, predicted, pos_label=positive),
                        "F1": f1_score(target, predicted, pos_label=positive),
                        "MCC": matthews_corrcoef(target, predicted),
                    }
                    actual = {metric.label: metric.value for metric in self.app.metric}
                    self.assertEqual(actual, {key: f"{value:.4f}" for key, value in expected.items()})
                    self.assertEqual(len(self.app.dataframe[0].value), 10)
                    self.check_classification_details(target, predicted)

    def check_classification_details(self, target, predicted):
        self.assertEqual(self.app.tabs[0].label, "Selected Model")
        tables = self.app.tabs[0].dataframe
        self.assertEqual(len(tables), 2)
        matrix = tables[0].value
        class_names = ["0 = Benign", "1 = Malignant"]
        self.assertEqual(list(matrix.index), class_names)
        self.assertEqual(list(matrix.columns), class_names)
        self.assertEqual(matrix.index.name, "Actual")
        self.assertEqual(matrix.columns.name, "Predicted")
        self.assertEqual(matrix.to_numpy().sum(), len(target))
        benign_correct, false_alarms, missed, detected = matrix.to_numpy().ravel()
        expected_summary = (
            f"Detected {detected} of {matrix.iloc[1].sum()} malignant cases; "
            f"missed {missed}; flagged {false_alarms} benign cases as malignant. "
            f"Correct predictions: {benign_correct + detected} of {len(target)}."
        )
        self.assertIn(expected_summary, [item.value for item in self.app.tabs[0].info])
        self.assertIsNotNone(self.heatmap)
        axis = self.heatmap.axes[0]
        self.assertEqual(axis.get_xlabel(), "Predicted diagnosis")
        self.assertEqual(axis.get_ylabel(), "Actual diagnosis")
        self.assertEqual(
            [text.get_text() for text in axis.texts],
            [
                f"{benign_correct}\nCorrectly benign",
                f"{false_alarms}\nFalse alarm",
                f"{missed}\nMissed malignant",
                f"{detected}\nCorrectly malignant",
            ],
        )
        self.assertEqual(
            [item.label for item in self.app.tabs[0].expander],
            [
                "How to read the confusion matrix",
                "View the exact matrix counts",
                "How to read the classification report",
            ],
        )
        for actual in (0, 1):
            for prediction in (0, 1):
                self.assertEqual(
                    matrix.iloc[actual, prediction],
                    ((target == actual) & (predicted == prediction)).sum(),
                )
        report = classification_report(
            target, predicted, labels=[0, 1], target_names=class_names,
            output_dict=True, zero_division=0,
        )
        expected = pd.DataFrame(
            {name: scores for name, scores in report.items() if name != "accuracy"}
        ).T
        pd.testing.assert_frame_equal(tables[1].value, expected)

    def test_invalid_uploads_show_errors_without_metrics(self):
        cases = [
            (self.data.drop(columns="target"), "0 = Benign, 1 = Malignant"),
            (self.data.drop(columns="mean radius"), "Missing feature columns"),
            (self.data.assign(target=2), "only 0 (Benign) or 1 (Malignant)"),
            (self.data.assign(target=float("nan")), "empty, NaN, or infinite"),
            (self.data.assign(**{"mean radius": float("inf")}), "infinite"),
            (self.data.assign(**{"mean radius": "invalid"}), "numeric"),
            (self.data.assign(**{"mean radius": " "}), "numeric"),
            (self.data.iloc[:0], "no data rows"),
        ]
        for data, message in cases:
            with self.subTest(message=message):
                self.upload(data)
                self.assertIn(message, self.app.error[0].value)
                self.assertEqual(len(self.app.metric), 0)
                self.assertEqual(len(self.app.tabs), 0)

    def test_single_class_has_no_auc(self):
        self.upload(self.data[self.data.target == 0])
        actual = {metric.label: metric.value for metric in self.app.metric}
        self.assertEqual(actual["AUC"], "N/A")
        self.assertTrue(any("only one class" in item.value for item in self.app.warning))
        self.assertEqual(len(actual), 6)
        subset = self.data[self.data.target == 0]
        model = joblib.load(ROOT / "model" / "logistic_regression.joblib")
        scaler = joblib.load(ROOT / "model" / "scaler.joblib")
        predicted = model.predict(scaler.transform(subset[list(scaler.feature_names_in_)]))
        self.check_classification_details(subset.target, predicted)
        self.assertTrue(any("Undefined per-class" in item.value for item in self.app.warning))

    def test_column_order_and_extra_columns_do_not_change_metrics(self):
        self.upload(self.data)
        expected = [metric.value for metric in self.app.metric]
        reordered = self.data[self.data.columns[::-1]].assign(notes="ignored")
        self.upload(reordered)
        self.assertEqual([metric.value for metric in self.app.metric], expected)


if __name__ == "__main__":
    unittest.main()
