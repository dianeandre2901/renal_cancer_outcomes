from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
# 
start_time = time.time()


# Load and select relevant columns
df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/df_clean.csv")
df = df.drop(columns=["event"])

df= df.head(50)
# Keep only rows with complete clinical info
df_clinical = df[["vital_status", "age_at_diagnosis_years", "tumour_grade", "tumour_stage"]].dropna()
# Encode target
df_clinical["vital_status"] = df_clinical["vital_status"].map({"Alive": 0, "Dead": 1})
# Features and labels
X = df_clinical[["age_at_diagnosis_years", "tumour_grade", "tumour_stage"]]
y = df_clinical["vital_status"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_scaled, y)

# Print coefficients
for feature, coef in zip(X.columns, clf.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# Predictions
y_pred = clf.predict(X_scaled)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))

# ROC Curve and AUC
y_prob = clf.predict_proba(X_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)



# Remove duplicate fpr values for interpolation
fpr_unique, unique_indices = np.unique(fpr, return_index=True)
tpr_unique = tpr[unique_indices]

# Now perform spline interpolation safely
fpr_new = np.linspace(0, 1, 200)
tpr_spline = make_interp_spline(fpr_unique, tpr_unique, k=2)
tpr_smooth = tpr_spline(fpr_new)
# Plot coefficients as a horizontal bar chart
plt.figure()
plt.plot(fpr_new, tpr_smooth, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# Plot coefficients
plt.bar(X.columns, clf.coef_[0])
plt.title("Logistic Regression Coefficients")
plt.xticks(rotation=45)


end_time = time.time()
elapsed = end_time - start_time

print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")


