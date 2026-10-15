#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------
# STEP 1: Load dataset
# -------------------------------------------------
df = pd.read_csv("synthetic_lung_disease_dataset.csv")

target_column = "Disease"

# Remove rows with missing target values
df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split BEFORE fitting imputers, SHAP, or MI
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# STEP 2: Handle missing values and categorical data
# -------------------------------------------------
numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = [
    c for c in X_train.columns if c not in numeric_cols
]

# Learn imputation values only from training data
if numeric_cols:
    num_imputer = SimpleImputer(strategy="median")

    X_train_num = pd.DataFrame(
        num_imputer.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols,
        index=X_train.index
    )
    X_test_num = pd.DataFrame(
        num_imputer.transform(X_test[numeric_cols]),
        columns=numeric_cols,
        index=X_test.index
    )
else:
    X_train_num = pd.DataFrame(index=X_train.index)
    X_test_num = pd.DataFrame(index=X_test.index)

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train[categorical_cols]),
        columns=categorical_cols,
        index=X_train.index
    )
    X_test_cat = pd.DataFrame(
        cat_imputer.transform(X_test[categorical_cols]),
        columns=categorical_cols,
        index=X_test.index
    )

    # One-hot encoding
    X_train_cat = pd.get_dummies(
        X_train_cat, dtype=int
    )
    X_test_cat = pd.get_dummies(
        X_test_cat, dtype=int
    )

    # Ensure identical columns in train and test
    X_train_cat, X_test_cat = X_train_cat.align(
        X_test_cat, join="left", axis=1, fill_value=0
    )
else:
    X_train_cat = pd.DataFrame(index=X_train.index)
    X_test_cat = pd.DataFrame(index=X_test.index)

X_train_processed = pd.concat(
    [X_train_num, X_train_cat], axis=1
)
X_test_processed = pd.concat(
    [X_test_num, X_test_cat], axis=1
)

# Consistent feature order
X_test_processed = X_test_processed[
    X_train_processed.columns
]

# -------------------------------------------------
# STEP 3: Calculate SHAP feature importance
# -------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_processed, y_train)

explainer = shap.TreeExplainer(rf)
shap_output = explainer(X_train_processed)

# Support binary and multiclass SHAP output formats
shap_values = shap_output.values

if isinstance(shap_values, list):
    # Older SHAP versions may return a list by class
    shap_importance = np.mean(
        np.abs(np.stack(shap_values, axis=-1)),
        axis=(0, 2)
    )
elif shap_values.ndim == 3:
    # Newer versions: samples x features x classes
    shap_importance = np.mean(
        np.abs(shap_values), axis=(0, 2)
    )
else:
    # Single-output classification
    shap_importance = np.mean(
        np.abs(shap_values), axis=0
    )

# -------------------------------------------------
# STEP 4: Calculate Mutual Information (MI)
# -------------------------------------------------
mi_scores = mutual_info_classif(
    X_train_processed,
    y_train,
    random_state=42
)

# -------------------------------------------------
# STEP 5: Normalize both scores
# -------------------------------------------------
def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=float)
    min_score = scores.min()
    max_score = scores.max()

    if np.isclose(max_score, min_score):
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


shap_normalized = minmax_normalize(shap_importance)
mi_normalized = minmax_normalize(mi_scores)

# -------------------------------------------------
# STEP 6: Combine SHAP and MI scores
# -------------------------------------------------
# Equal weights are a starting choice, not a universal optimum
alpha = 0.5

combined_score = (
    alpha * shap_normalized
    + (1 - alpha) * mi_normalized
)

feature_ranking = pd.DataFrame({
    "Feature": X_train_processed.columns,
    "SHAP_Importance": shap_importance,
    "MI_Score": mi_scores,
    "Normalized_SHAP": shap_normalized,
    "Normalized_MI": mi_normalized,
    "SHAP_MI_Score": combined_score
}).sort_values(
    "SHAP_MI_Score", ascending=False
).reset_index(drop=True)

# -------------------------------------------------
# STEP 7: Select top-k features
# -------------------------------------------------
# Change k based on your research requirement
k = min(10, X_train_processed.shape[1])

selected_features = feature_ranking[
    "Feature"
].head(k).tolist()

print("\nSHAP-MI Feature Ranking:")
print(feature_ranking.to_string(index=False))

print("\nSelected Features:")
print(selected_features)

# Save the feature-ranking results
feature_ranking.to_csv(
    "SHAP_MI_feature_ranking.csv", index=False
)

# -------------------------------------------------
# STEP 8: Train SVM using selected features
# -------------------------------------------------
X_train_selected = X_train_processed[selected_features]
X_test_selected = X_test_processed[selected_features]

svm_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
)

svm_model.fit(X_train_selected, y_train)
y_pred = svm_model.predict(X_test_selected)

print("\nSVM Results Using SHAP-MI Features")
print("Selected feature count:", len(selected_features))
print("Test accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_.astype(str),
    zero_division=0
))

# -------------------------------------------------
# STEP 9: Plot the combined feature scores
# -------------------------------------------------
top_features = feature_ranking.head(k).sort_values(
    "SHAP_MI_Score"
)

plt.figure(figsize=(9, 6))
plt.barh(
    top_features["Feature"],
    top_features["SHAP_MI_Score"]
)
plt.xlabel("Normalized combined SHAP-MI score")
plt.ylabel("Feature")
plt.title("SHAP-MI Feature Selection")
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import pandas as pd

from itertools import combinations
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTENC


# =========================================================
# STEP 1: Load dataset
# =========================================================
df = pd.read_csv("synthetic_lung_disease_dataset.csv")

target_column = "Disease"
df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column])
y = df[target_column].astype(str)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())


# =========================================================
# STEP 2: Split before fitting preprocessing
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining class distribution:")
print(y_train.value_counts())


# =========================================================
# STEP 3: Separate numerical and categorical features
# =========================================================
numeric_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = [
    c for c in X_train.columns if c not in numeric_cols
]

if not numeric_cols:
    raise ValueError("At least one numeric feature is required.")

if X_train.empty:
    raise ValueError("Training dataset is empty.")


# =========================================================
# STEP 4: Impute missing values
# =========================================================
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

Xtr_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index
)

Xte_num = pd.DataFrame(
    num_imputer.transform(X_test[numeric_cols]),
    columns=numeric_cols,
    index=X_test.index
)

if categorical_cols:
    Xtr_cat_raw = pd.DataFrame(
        cat_imputer.fit_transform(X_train[categorical_cols]),
        columns=categorical_cols,
        index=X_train.index
    )

    Xte_cat_raw = pd.DataFrame(
        cat_imputer.transform(X_test[categorical_cols]),
        columns=categorical_cols,
        index=X_test.index
    )
else:
    Xtr_cat_raw = pd.DataFrame(index=X_train.index)
    Xte_cat_raw = pd.DataFrame(index=X_test.index)


# =========================================================
# STEP 5: Prepare numeric and categorical features
# =========================================================
# Scale numeric values before SMOTE so distance calculations
# are not dominated by features with large numerical ranges.
numeric_scaler = StandardScaler()

Xtr_num_scaled = numeric_scaler.fit_transform(Xtr_num)
Xte_num_scaled = numeric_scaler.transform(Xte_num)

if categorical_cols:
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    Xtr_cat_ord = ordinal_encoder.fit_transform(Xtr_cat_raw)
    Xte_cat_ord = ordinal_encoder.transform(Xte_cat_raw)

    # SMOTENC requires categorical feature indices.
    n_num = len(numeric_cols)
    cat_indices = list(
        range(n_num, n_num + len(categorical_cols))
    )

    Xtr_for_sampling = np.column_stack([
        Xtr_num_scaled,
        Xtr_cat_ord
    ])

    Xte_for_sampling = np.column_stack([
        Xte_num_scaled,
        Xte_cat_ord
    ])
else:
    Xtr_for_sampling = Xtr_num_scaled.copy()
    Xte_for_sampling = Xte_num_scaled.copy()
    cat_indices = []


# =========================================================
# STEP 6: Adaptive SMOTE
# =========================================================
class_counts = Counter(y_train)
print("\nBefore adaptive SMOTE:", class_counts)

if len(class_counts) < 2:
    raise ValueError("At least two classes are required.")

if len(set(class_counts.values())) == 1:
    # Balanced data: no synthetic samples are necessary.
    X_resampled = Xtr_for_sampling.copy()
    y_resampled = y_train.to_numpy().copy()

    print("Dataset is balanced. SMOTE skipped.")

else:
    min_count = min(class_counts.values())

    if min_count < 2:
        raise ValueError(
            "SMOTE needs at least two training samples "
            "in every class to be oversampled. Collect more "
            "samples or use a suitable alternative."
        )

    # Adaptive neighbor count based on the smallest class.
    k_neighbors = min(5, min_count - 1)

    # Oversample each minority class to the majority count.
    target_count = max(class_counts.values())

    sampling_strategy = {
        cls: target_count
        for cls, count in class_counts.items()
        if count < target_count
    }

    if categorical_cols:
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )
    else:
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )

    X_resampled, y_resampled = sampler.fit_resample(
        Xtr_for_sampling, y_train
    )

print("After adaptive SMOTE:", Counter(y_resampled))


# =========================================================
# STEP 7: One-hot encode categorical features for SVM
# =========================================================
if categorical_cols:
    n_num = len(numeric_cols)

    # Fit the encoder on resampled training categories.
    onehot = OneHotEncoder(handle_unknown="ignore")

    X_cat_resampled = X_resampled[:, n_num:]
    X_cat_test = Xte_for_sampling[:, n_num:]

    X_cat_resampled = onehot.fit_transform(
        X_cat_resampled
    ).toarray()

    X_cat_test = onehot.transform(
        X_cat_test
    ).toarray()

    X_svm_train = np.column_stack([
        X_resampled[:, :n_num],
        X_cat_resampled
    ])

    X_svm_test = np.column_stack([
        Xte_for_sampling[:, :n_num],
        X_cat_test
    ])
else:
    X_svm_train = X_resampled
    X_svm_test = Xte_for_sampling


# =========================================================
# STEP 8: Create multiple binary SVM submodels
# =========================================================
classes = np.unique(y_resampled)

if len(classes) < 2:
    raise ValueError("At least two classes are required.")

svm_models = {}
pairs = list(combinations(classes, 2))

print("\nNumber of classes:", len(classes))
print("Number of SVM submodels:", len(pairs))

for class_a, class_b in pairs:

    # Select training samples belonging to this pair only.
    mask = np.isin(y_resampled, [class_a, class_b])

    X_pair = X_svm_train[mask]
    y_pair = y_resampled[mask]

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    )

    svm.fit(X_pair, y_pair)

    svm_models[(class_a, class_b)] = svm

    print(f"Trained submodel: {class_a} vs {class_b}")


# =========================================================
# STEP 9: Multiclass voting
# =========================================================
def predict_multiclass(X_input, models, class_labels):
    predictions = []

    for row in X_input:
        sample = row.reshape(1, -1)

        votes = {c: 0 for c in class_labels}
        margins = {c: 0.0 for c in class_labels}

        for (class_a, class_b), svm in models.items():
            prediction = svm.predict(sample)[0]
            margin = float(svm.decision_function(sample)[0])

            votes[prediction] += 1

            # SVC binary decision_function is positive toward
            # classes_[1], which is class_b for this pair.
            margins[class_b] += margin
            margins[class_a] -= margin

        # First maximize votes; use aggregate margins to break ties.
        winner = max(
            class_labels,
            key=lambda c: (votes[c], margins[c])
        )

        predictions.append(winner)

    return np.asarray(predictions)


y_pred = predict_multiclass(
    X_svm_test,
    svm_models,
    classes
)


# =========================================================
# STEP 10: Evaluation
# =========================================================
print("\n========== ADAPTIVE SMOTE + MULTI-SVM ==========")

print("Accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, y_pred), 4))

print("Macro precision:",
      round(precision_score(
          y_test, y_pred, average="macro", zero_division=0
      ), 4))

print("Macro recall:",
      round(recall_score(
          y_test, y_pred, average="macro", zero_division=0
      ), 4))

print("Macro F1:",
      round(f1_score(
          y_test, y_pred, average="macro", zero_division=0
      ), 4))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test, y_pred, labels=classes
))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=classes,
    zero_division=0
))


# =========================================================
# STEP 11: Save results
# =========================================================
results = X_test.copy()
results["Actual_Disease"] = y_test
results["Predicted_Disease"] = y_pred

results.to_csv(
    "adaptive_smote_multi_svm_results.csv",
    index=False
)

print("\nSaved: adaptive_smote_multi_svm_results.csv")

