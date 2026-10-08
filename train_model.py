import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("diabetes_risk.csv")

print("========================================")
print("DATASET")
print("========================================")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ==========================================================
# REMOVE PATIENT ID
# ==========================================================

df = df.drop("patient_id", axis=1)


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop("diabetes_risk", axis=1)
y = df["diabetes_risk"]


# ==========================================================
# ENCODE TARGET
# ==========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nTarget classes:")
print(label_encoder.classes_)


# ==========================================================
# FIND NUMERICAL AND CATEGORICAL COLUMNS
# ==========================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns


# ==========================================================
# PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),

            numerical_columns
        ),

        (
            "categorical",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),

            categorical_columns
        )
    ]
)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ==========================================================
# SVM
# ==========================================================

print("\nTraining SVM...")

svm_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        SVC(kernel="rbf")
    )
])

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

print("SVM Accuracy:", svm_accuracy)


# ==========================================================
# DECISION TREE
# ==========================================================

print("\nTraining Decision Tree...")

decision_tree_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_pred = decision_tree_model.predict(
    X_test
)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

print(
    "Decision Tree Accuracy:",
    decision_tree_accuracy
)


# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest...")

random_forest_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_pred = random_forest_model.predict(
    X_test
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

print(
    "Random Forest Accuracy:",
    random_forest_accuracy
)


# ==========================================================
# KNN
# ==========================================================

print("\nTraining KNN...")

knn_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        KNeighborsClassifier(
            n_neighbors=5
        )
    )
])

knn_model.fit(
    X_train,
    y_train
)

knn_pred = knn_model.predict(
    X_test
)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

print(
    "KNN Accuracy:",
    knn_accuracy
)


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression...")

logistic_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

logistic_model.fit(
    X_train,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

print(
    "Logistic Regression Accuracy:",
    logistic_accuracy
)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = {

    "SVM": svm_accuracy,

    "Decision Tree": decision_tree_accuracy,

    "Random Forest": random_forest_accuracy,

    "KNN": knn_accuracy,

    "Logistic Regression": logistic_accuracy

}


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

for name, accuracy in results.items():

    print(
        name,
        ":",
        round(accuracy * 100, 2),
        "%"
    )


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

best_model_name = max(
    results,
    key=results.get
)


if best_model_name == "SVM":

    best_model = svm_model

elif best_model_name == "Decision Tree":

    best_model = decision_tree_model

elif best_model_name == "Random Forest":

    best_model = random_forest_model

elif best_model_name == "KNN":

    best_model = knn_model

else:

    best_model = logistic_model


print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Best Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(
        results[best_model_name] * 100,
        2
    ),
    "%"
)


# ==========================================================
# SAVE EVERYTHING INTO PICKLE
# ==========================================================

model_data = {

    "model": best_model,

    "label_encoder": label_encoder,

    "model_name": best_model_name,

    "accuracy": results[best_model_name]

}


with open(
    "diabetes_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


print("\n========================================")
print("DONE")
print("========================================")

print(
    "diabetes_model.pkl created successfully!"
)