import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import openslide
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import time 

start_time = time.time()

# === DEVICE CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === DATASET CLASS ===

class PrecomputedFeatureSurvivalDataset(Dataset):
    def __init__(self, slide_ids, clinical_df, feature_dict):
        clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)
        self.clinical_df = clinical_df.set_index("slide_id")
        self.slide_ids = [str(sid) for sid in slide_ids]
        self.feature_dict = feature_dict
        self.default_zero_tensor = np.zeros((1, 512), dtype=np.float32)
        self.tabular_data = clinical_df.set_index("slide_id")[["age_at_diagnosis_years", "tumour_stage", "tumour_grade"]]

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = str(self.slide_ids[idx])
        features = self.feature_dict.get(slide_id, self.default_zero_tensor)
        features = torch.tensor(features).float()
        if slide_id not in self.clinical_df.index:
            raise ValueError(f"Slide ID '{slide_id}' not found in clinical_df index: {self.clinical_df.index[:5]}")
        time = torch.tensor(self.clinical_df.loc[slide_id, "os_days"]).float()
        event = torch.tensor(self.clinical_df.loc[slide_id, "event"]).float()
        tabular = self.tabular_data.loc[slide_id].values.astype(np.float32)
        tabular = torch.tensor(tabular)
        return features, time, event, tabular


# === SURFORMER BACKBONE ===

class SurformerMIL(nn.Module):
    def __init__(self, dim=512, heads=8, layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, dim))
        self.patch_embed = nn.Linear(dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output = nn.Sequential(
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if x.size(1) > self.pos_embed.size(1):
            repeat_times = (x.size(1) // self.pos_embed.size(1)) + 1
            extended_pos_embed = self.pos_embed.repeat(1, repeat_times, 1)[:, :x.size(1), :]
            x = x + extended_pos_embed
        else:
            x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x[:, 0])  # (B, 512)

# === FINAL MODEL ===

class WSI_SurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.surformer = SurformerMIL()
        self.tabular_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(512 + 128, 1)

    def forward(self, x, tabular):
        surformer_feat = self.surformer(x).squeeze(-1)  # shape (B, 512)
        tabular_feat = self.tabular_net(tabular)
        combined = torch.cat([surformer_feat, tabular_feat], dim=1)
        return self.final_fc(combined).squeeze(-1)

# === COX LOSS ===
def cox_ph_loss(risk, time, event):
    if event.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk.device)
    hazard_diff = risk.unsqueeze(1) - risk.unsqueeze(0)
    time_order = (time.unsqueeze(0) >= time.unsqueeze(1)).float()
    log_risk = torch.logsumexp(hazard_diff * time_order, dim=1)
    loss = -torch.sum((risk - log_risk) * event) / event.sum()
    return loss


# === TRAINING FUNCTION ===

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for x, time, event, tabular in dataloader:
        if torch.isnan(x).any() or torch.isnan(time).any() or torch.isnan(event).any() or torch.isnan(tabular).any():
            print("NaNs found in input, skipping batch.")
            continue
        if x.numel() == 0:
            print("Empty input tensor, skipping batch.")
            continue
        x, time, event, tabular = x.to(device), time.to(device), event.to(device), tabular.to(device)
        risk = model(x, tabular)
        loss = cox_ph_loss(risk, time, event)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch_loss(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, time, event, tabular in dataloader:
            if x.numel() == 0 or torch.isnan(x).any() or torch.isnan(time).any() or torch.isnan(event).any() or torch.isnan(tabular).any():
                continue
            x, time, event, tabular = x.to(device), time.to(device), event.to(device), tabular.to(device)
            risk = model(x, tabular)
            loss = cox_ph_loss(risk, time, event)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('nan')

# === EVAL FUNCTION ===

def eval_epoch(model, dataloader):
    model.eval()
    risks, times, events = [], [], []

    with torch.no_grad():
        for x, time, event, tabular in dataloader:
            if x.numel() == 0:
                print("Empty input tensor in eval, skipping.")
                continue
            if torch.isnan(x).any() or torch.isnan(tabular).any():
                print("NaNs in eval input, skipping.")
                continue
            x = x.to(device)
            tabular = tabular.to(device)
            risk = model(x, tabular).cpu().numpy()
            risks.extend(risk)
            times.extend(time.numpy())
            events.extend(event.numpy())

    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events)

    mask = ~np.isnan(risks) & ~np.isnan(times) & ~np.isnan(events)
    if mask.sum() == 0:
        print("Warning: No valid samples for concordance index.")
        return float("nan")

    risks = risks[mask]
    times = times[mask]
    events = events[mask]

    return concordance_index(times, -risks, events)

# === MAIN LOOP ===

if __name__ == "__main__":
    clinical_train_df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv", dtype=str)
    clinical_val_df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv", dtype=str)
    # Preload features into memory for fast access

    clinical_train_df["os_days"] = pd.to_numeric(clinical_train_df["os_days"], errors="coerce")
    clinical_train_df["event"] = pd.to_numeric(clinical_train_df["event"], errors="coerce")

    clinical_val_df["os_days"] = pd.to_numeric(clinical_val_df["os_days"], errors="coerce")
    clinical_val_df["event"] = pd.to_numeric(clinical_val_df["event"], errors="coerce")
    feature_dict_train = {}
    feature_dir_train = "/rds/general/user/dla24/home/thesis/src/scripts/features_train"
    for fname in os.listdir(feature_dir_train):
        if fname.endswith(".npy"):
            sid = fname.replace(".npy", "")
            try:
                feature_dict_train[sid] = np.load(os.path.join(feature_dir_train, fname))
            except Exception as e:
                print(f"Failed to load features for {sid}: {e}")
                feature_dict_train[sid] = np.zeros((1, 512), dtype=np.float32)

    feature_dict_val = {}
    feature_dir_val = "/rds/general/user/dla24/home/thesis/src/scripts/features_train_validation"
    for fname in os.listdir(feature_dir_val):
        if fname.endswith(".npy"):
            sid = fname.replace(".npy", "")
            try:
                feature_dict_val[sid] = np.load(os.path.join(feature_dir_val, fname))
            except Exception as e:
                print(f"Failed to load features for {sid}: {e}")
                feature_dict_val[sid] = np.zeros((1, 512), dtype=np.float32)

    # Filter clinical dataframes by intersection with features
    train_slide_mask = clinical_train_df["slide_id"].isin(feature_dict_train.keys())
    clinical_train_df = clinical_train_df[train_slide_mask].reset_index(drop=True)
    # Balanced selection: ensure mix of censored and uncensored slides
    uncensored_df = clinical_train_df[clinical_train_df["event"] == 1]
    censored_df = clinical_train_df[clinical_train_df["event"] == 0]

    uncensored_ids = uncensored_df["slide_id"].unique()[:20]
    censored_ids = censored_df["slide_id"].unique()[:20]
    train_slide_ids = np.concatenate([uncensored_ids, censored_ids])

    val_slide_mask = clinical_val_df["slide_id"].isin(feature_dict_val.keys())
    clinical_val_df = clinical_val_df[val_slide_mask].reset_index(drop=True)
    val_slide_ids = clinical_val_df["slide_id"].unique()[:30]

    # Check for presence of events in training set
    train_events = clinical_train_df[clinical_train_df["slide_id"].isin(train_slide_ids)]["event"]
    if train_events.sum() == 0:
        print("Warning: No slides with event==1 in training data. Exiting.")
        exit(1)

    train_data = PrecomputedFeatureSurvivalDataset(train_slide_ids, clinical_train_df, feature_dict_train)
    val_data = PrecomputedFeatureSurvivalDataset(val_slide_ids, clinical_val_df, feature_dict_val)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

    x_sample, time_sample, event_sample, tab_sample = next(iter(train_loader))
    print("Sample input shape:", x_sample.shape)
    print("Sample tabular:", tab_sample)

    model = WSI_SurvivalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_losses = []
    val_losses = []
    val_cindices = []

    best_val_cindex = 0
    patience = 5
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, 21):
        loss = train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch_loss(model, val_loader)
        cindex = eval_epoch(model, val_loader)
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | C-Index: {cindex:.4f}")
        train_losses.append(loss)
        val_losses.append(val_loss)
        val_cindices.append(cindex)

        if cindex > best_val_cindex:
            best_val_cindex = cindex
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

# === After Training: Plotting ===
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curve")
    plt.legend()
    plt.savefig("results/plots/surformer_train_val_loss_curve.png")

    plt.figure()
    plt.plot(val_cindices, label="Validation C-Index")
    plt.xlabel("Epoch")
    plt.ylabel("C-Index")
    plt.title("Validation Concordance Index")
    plt.legend()
    plt.savefig("val_cindex_curve.png")

# === ROC-like C-statistic visualization ===
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for x, time, event, tabular in val_loader:
            if x.numel() == 0:
                print("Empty input tensor in eval, skipping.")
                continue
            if torch.isnan(x).any() or torch.isnan(tabular).any():
                print("NaNs in eval input, skipping.")
                continue
            x = x.to(device)
            tabular = tabular.to(device)
            risk = model(x, tabular).cpu().numpy()
            risks.extend(risk)
            times.extend(time.numpy())
            events.extend(event.numpy())

    risks = np.array(risks)
    times = np.array(times)
    events = np.array(events)

    mask = ~np.isnan(risks) & ~np.isnan(times) & ~np.isnan(events)
    risks = risks[mask]
    times = times[mask]
    events = events[mask]

    # Sort by decreasing risk
    sorted_indices = np.argsort(-risks)
    risks_sorted = risks[sorted_indices]
    times_sorted = times[sorted_indices]
    events_sorted = events[sorted_indices]

    # Normalize survival times to [0,1]
    times_norm = (times_sorted - times_sorted.min()) / (times_sorted.max() - times_sorted.min() + 1e-8)


# === Performance Plot: Final C-index Overlay on ROC-style visual ===
    plt.figure(figsize=(10, 4))
    plt.plot(times_norm, label="Normalized Survival Time", color='blue', lw=2)
    plt.step(range(len(events_sorted)), events_sorted, where='mid', label="Event (1=death,0=censored)", color='orange', lw=2)
    plt.xlabel("Samples sorted by decreasing risk")
    plt.ylabel("Value")
    plt.title(f"ROC-like C-statistic Visualization (Final C-index = {best_val_cindex:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/surformer_final_cstat_curve_with_cindex.pdf")
    plt.close()

from sklearn.metrics import roc_curve, auc

# Optional: approximate binary label at a time threshold (e.g. survival > 1000 days = 0, else 1)
TIME_THRESHOLD = 1000

binary_labels = (times < TIME_THRESHOLD) & (events == 1)  # Only uncensored deaths
binary_labels = binary_labels.astype(int)

valid_mask = ~np.isnan(risks) & ~np.isnan(binary_labels)
risks_binary = -risks[valid_mask]  # higher risk = more likely to die before threshold
labels_binary = binary_labels[valid_mask]

if len(np.unique(labels_binary)) == 2:
    fpr, tpr, _ = roc_curve(labels_binary, risks_binary)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (threshold: {TIME_THRESHOLD} days)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/surformer_binary_auc_proxy.png")
    plt.close()
else:
    print("Skipped ROC curve: only one class in binary label proxy.")

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
with open("results/plots/surformer_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
