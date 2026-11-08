# Breast Cancer Wisconsin Classification

## Problem Statement

This project compares five supervised learning classifiers on the Breast Cancer Wisconsin (Diagnostic) data set. Each model predicts whether the tumor is malignant or benign based on measurements computed from a digitized fine needle aspirate image.

## Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository, [dataset 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B)
- **Access method:** scikit-learn's bundled copy (`load_breast_cancer(as_frame=True)`); no download is needed at runtime.
- **Instances:** 569
- **Predictive features:** 30 numeric measurements
- **Target:** `target`, encoded as `1` = malignant and `0` = benign

The features describe ten properties of cell-nuclei such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each property is given as the mean, standard error and the worst value. The display-only diagnosis label is not included in model inputs.

### Preprocessing and spliting

The scikit-learn target labels are flipped so that malignant is the positive class (`1`). A stratified 80/20 train-test split is used with `random_state=42`. `StandardScaler` is fit on training features only and reused to transform test data and uploaded CSVs. All five models use the same splitting and preprocessing, and the held-out test data is in [`test_data.csv`](test_data.csv).

## Models and Evaluation

All metrics below were calculated on the saved notebook evaluation run on the same held-out test split. Precision, recall, F1, and AUC use malignant (target=1) as the positive class. AUC uses predicted probabilities where available and for SVM decision scores.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.964912 | 0.996032 | 0.975000 | 0.928571 | 0.951220 | 0.924518 |
| Decision Tree | 0.929825 | 0.924603 | 0.904762 | 0.904762 | 0.904762 | 0.849206 |
| KNN | 0.956140 | 0.982308 | 0.974359 | 0.904762 | 0.938272 | 0.905824 |
| Naive Bayes (Gaussian) | 0.921053 | 0.989087 | 0.923077 | 0.857143 | 0.888889 | 0.829162 |
| Support Vector Machine | 0.973684 | 0.994709 | 1.000000 | 0.928571 | 0.962963 | 0.944155 |

### Performance observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Good results overall, 0.964912 accuracy and 0.951220 F1. It achieved the best AUC (0.996032). |
| Decision Tree | Accuracy 0.929825, F1 0.904762. This comparison had the lowest AUC (0.924603). |
| KNN | Accuracy: 0.956140, F1: 0.938272, Malignant Recall: 0.904762. |
| Naive Bayes | Got the lowest accuracy (0.921053) and malignant recall (0.857143) but still had a high AUC of 0.989087. |
| Support Vector Machine | Highest in accuracy (0.973684), precision (1.000000), F1 (0.962963), MCC (0.944155). |
| Overall, Winner for your dataset? | Winner for this split: Support Vector Machine. It was top on four of the six metrics, with Logistic Regression having the highest AUC. This comparison is specific to the split and settings selected and is not proof that one model will be best on other data. |

## Streamlit Application

The app loads the saved classifiers and scaler from `model/`, accepts a labeled test CSV, validates the required columns and values, and reports Accuracy, AUC, Precision, Recall, F1, MCC, a confusion matrix, and a classification report. It does not retrain models on uploaded data.

**Live app:** [Open the Breast Cancer Model Comparison app](https://breast-cancer-model-comparison-ubzg3akmc67gg6pgwxjut8.streamlit.app/)

Run locally from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload [`test_data.csv`](test_data.csv) to evaluate the saved models. To run the app tests:

```powershell
python -m unittest discover -s tests
```

## Repository

[GitHub repository](https://github.com/vsekar234/breast-cancer-model-comparison)

The repository includes [`app.py`](app.py), the evaluation notebook, [`requirements.txt`](requirements.txt), the held-out test CSV, and the five saved classifiers and scaler under [`model/`](model/).
