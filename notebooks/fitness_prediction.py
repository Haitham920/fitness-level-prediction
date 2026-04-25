# ============================================================
# FITNESS LEVEL PREDICTION
# Models: Random Forest + HistGradientBoosting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/fitness_dataset.csv"

# ============================================================
# DATA LOADING & SAMPLING STRATEGY 
# ============================================================
SAMPLE_SIZE = 250_000

df_full = pd.read_csv(DATA_PATH)

df, _ = train_test_split(
    df_full,
    train_size=SAMPLE_SIZE,
    stratify=df_full["fitness_level"],
    random_state=RANDOM_STATE
)

df = df.reset_index(drop=True)

print(df.shape)
print(df["fitness_level"].value_counts())
# ============================================================
# EDA 
# ============================================================

# CLASS DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="fitness_level", order=df["fitness_level"].value_counts().index)
plt.title("Class Distribution")
plt.show()

# HISTOGRAMS
df.select_dtypes(include=np.number).hist(figsize=(15,10), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

# BOXPLOTS
key_features = [
    "steps_per_day", "bmi", "sleep_hours", "stress_level",
    "active_minutes", "workouts_per_week",
    "consistency_score", "heart_rate_resting"
]

plt.figure(figsize=(16,10))
for i, col in enumerate(key_features, 1):
    plt.subplot(2,4,i)
    sns.boxplot(data=df, x="fitness_level", y=col)
    plt.title(col)
plt.tight_layout()
plt.show()

# CORRELATION HEATMAP
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# RELATIONSHIP PLOTS
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x="fitness_level", y="steps_per_day")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="fitness_level", y="bmi")
plt.show()
# ============================================================
# FEATURE ENGINEERING
# Composite Fitness Score
# ============================================================

def minmax(col):
    return (col - col.min()) / (col.max() - col.min())

df["fitness_score"] = (
    0.25 * minmax(df["steps_per_day"]) +
    0.20 * minmax(df["active_minutes"]) +
    0.15 * (df["sleep_quality"] / 10) +
    0.15 * (df["diet_quality"] / 10) +
    0.10 * (df["workouts_per_week"] / 7) +
    0.10 * (1 - minmax(df["sedentary_minutes"])) +   # inverted
    0.10 * (1 - df["stress_level"] / 10)             # inverted
)

print("\nfitness_score added.")
print(df["fitness_score"].describe())
# ============================================================
# 5. PREPROCESSING 
# ============================================================

df = df.drop(columns=["calories_burned"])

le = LabelEncoder()
df["lifestyle_category"] = le.fit_transform(df["lifestyle_category"])

target_map = {"Unfit":0, "Moderate":1, "Fit":2}
df["target"] = df["fitness_level"].map(target_map)

X = df.drop(columns=["fitness_level", "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 6. SMOTE
# ============================================================

def simple_smote(X, y, minority_class=None):
    X = np.array(X)
    y = np.array(y)

    # auto-detect minority class (better than hardcoding class 2)
    if minority_class is None:
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]

    X_min = X[y == minority_class]

    n_synth = len(X) - 2 * len(X_min)

    if n_synth <= 0:
        print("No synthetic samples needed.")
        return X, y

    synth = []

    for _ in range(n_synth):
        i, j = np.random.randint(0, len(X_min), 2)
        lam = np.random.rand()

        synth.append(
            X_min[i] + lam * (X_min[j] - X_min[i])
        )

    X_new = np.vstack([X, synth])
    y_new = np.hstack(
        [y, np.full(len(synth), minority_class)]
    )

    return X_new, y_new


X_train_sm, y_train_sm = simple_smote(X_train, y_train)

print("Original:", X_train.shape)
print("After SMOTE:", X_train_sm.shape)


# ============================================================
# 7. MODELS (RF + HistGradientBoosting)
# ============================================================

def evaluate(model, name, use_smote=False):

    if use_smote:
        model.fit(X_train_sm, y_train_sm)
    else:
        model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\n" + "="*50)
    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Macro:", f1_score(y_test, pred, average="macro"))
    print(classification_report(y_test, pred))


# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

evaluate(rf, "Random Forest (Without SMOTE)")
evaluate(rf, "Random Forest (With SMOTE)", use_smote=True)


# Histogram Gradient Boosting (NOT LightGBM)
hgb = HistGradientBoostingClassifier(
    max_depth=7,
    learning_rate=0.1,
    max_iter=200,
    random_state=RANDOM_STATE
)

evaluate(hgb, "HistGradientBoosting (Without SMOTE)")
evaluate(hgb, "HistGradientBoosting (With SMOTE)", use_smote=True)


# ============================================================
# 8. CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

models = {
    "Random Forest": rf,
    "HistGradientBoosting": hgb
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(
        f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}"
    )


# ============================================================
# 9. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[10,15,20],
    "min_samples_leaf":[5,10,20]
}

rf_model = RandomForestClassifier(
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=6,
    cv=2,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

# tune on SMOTE-balanced data
search.fit(X_train_sm, y_train_sm)

best_rf = search.best_estimator_

pred = best_rf.predict(X_test)

print("\nBest params:", search.best_params_)
print(
    "Tuned RF F1:",
    f1_score(y_test, pred, average="macro")
)


# ============================================================
# 10. LEAKAGE / FEATURE IMPORTANCE
# ============================================================

rf_full = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

rf_full.fit(X_train_sm, y_train_sm)

imp = pd.Series(
    rf_full.feature_importances_,
    index=X.columns
)

imp_no_score = (
    imp.drop(
        labels=["fitness_score"],
        errors="ignore"
    )
    .sort_values(ascending=False)
)

print("Top features excluding fitness_score:")
print(imp_no_score.head(10))


plt.figure(figsize=(8,5))
imp_no_score.head(10).plot(kind="barh")
plt.title("Feature Importance (Leakage Check)")

plt.savefig(
    "feature_importance.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL MODEL
# ============================================================

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

final_model.fit(X_train_sm, y_train_sm)

final_pred = final_model.predict(X_test)

print("\nFINAL MODEL RESULTS")
print(
    "Accuracy:",
    accuracy_score(y_test, final_pred)
)

print(
    "F1 Macro:",
    f1_score(y_test, final_pred, average="macro")
)

print(
    classification_report(
        y_test,
        final_pred
    )
)


# ============================================================
# MODEL COMPARISON VISUAL
# ============================================================

results = pd.DataFrame({
    "Model":[
        "Random Forest",
        "HistGradientBoosting"
    ],
    "F1 Score":[
        f1_score(
            y_test,
            rf.predict(X_test),
            average="macro"
        ),
        f1_score(
            y_test,
            hgb.predict(X_test),
            average="macro"
        )
    ]
})

plt.figure(figsize=(6,4))

sns.barplot(
    data=results,
    x="Model",
    y="F1 Score"
)

plt.title("Model Comparison (F1 Macro)")
plt.ylim(0,1)

plt.savefig(
    "model_comparison.png",
    bbox_inches="tight"
)

plt.show()

print(results)