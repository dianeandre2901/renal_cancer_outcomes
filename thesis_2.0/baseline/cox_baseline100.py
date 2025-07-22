import pandas as pd
from lifelines import CoxPHFitter
import time
import matplotlib.pyplot as plt
# 
start_time = time.time()


# Load and select relevant columns
df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/df_clean.csv")
df = df.drop(columns=["vital_status"])

df= df.head(100)
# Keep only rows with complete clinical info
df_clinical = df[["os_days", "event", "age_at_diagnosis_years", "tumour_grade", "tumour_stage"]].dropna()
cph = CoxPHFitter()
cph.fit(df_clinical, duration_col="os_days", event_col="event")
cph.print_summary()


cph.plot()
plt.title("Cox Model Coefficients")
plt.tight_layout()
plt.show()

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
