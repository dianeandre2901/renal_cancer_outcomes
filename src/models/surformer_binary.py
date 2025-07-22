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

class PrecomputedFeatureBinaryDataset(Dataset):
    def __init__(self, slide_ids, clinical_df, feature_dict):
        clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)
        self.clinical_df = clinical_df.set_index("slide_id")
        self.slide_ids = [str(sid) for sid in slide_ids]
        self.feature_dict = feature_dict
        self.default_zero_tensor = np.zeros((1, 512), dtype=np.float32)

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = str(self.slide_ids[idx])
        features = self.feature_dict.get(slide_id, self.default_zero_tensor)
        features = torch.tensor(features).float()
        if slide_id not in self.clinical_df.index:
            raise ValueError(f"Slide ID '{slide_id}' not found in clinical_df index.")
        label = torch.tensor(self.clinical_df.loc[slide_id, "vital_status"]).float()
        return features, label


# === SURFORMER BACKBONE ===

class SurformerMIL(nn.Module):
    def __init__(self, dim=512, heads=8, layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, dim))
        self.patch_embed = nn.Linear(dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Dropout(0.3),  
        nn.Linear(dim, 1)
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
        return self.output(x[:, 0])  # return logit score

# === FINAL MODEL ===

class WSI_BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.surformer = SurformerMIL()

    def forward(self, x):
        # x: (B, N, D) where D is feature dimension
        return self.surformer(x).squeeze(-1)

# === TRAINING FUNCTION ===


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, label in dataloader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == label).sum().item()
        total += label.size(0)

    acc = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), acc

# === EVAL FUNCTION ===

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, label in dataloader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criterion(logits, label)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

    acc = correct / total if total > 0 else 0.0
    return total_loss / len(dataloader), acc

# === MAIN LOOP ===

if __name__ == "__main__":
    clinical_train_df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/train_40x.csv", dtype=str)
    clinical_val_df = pd.read_csv("/rds/general/user/dla24/home/thesis/TGCA_dataset/val_40x.csv", dtype=str)
    # Preload features into memory for fast access

    label_map = {"Alive": 0, "Dead": 1}
    clinical_train_df["vital_status"] = clinical_train_df["vital_status"].map(label_map)
    clinical_val_df["vital_status"] = clinical_val_df["vital_status"].map(label_map)

    clinical_train_df = clinical_train_df.dropna(subset=["vital_status"])
    clinical_val_df = clinical_val_df.dropna(subset=["vital_status"])

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
    train_slide_ids = clinical_train_df["slide_id"].unique()[:70]

    val_slide_mask = clinical_val_df["slide_id"].isin(feature_dict_val.keys())
    clinical_val_df = clinical_val_df[val_slide_mask].reset_index(drop=True)
    val_slide_ids = clinical_val_df["slide_id"].unique()[:50]

    train_data = PrecomputedFeatureBinaryDataset(train_slide_ids, clinical_train_df, feature_dict_train)
    val_data = PrecomputedFeatureBinaryDataset(val_slide_ids, clinical_val_df, feature_dict_val)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

    model = WSI_BinaryModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, 21):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

# === Post-training Plots ===
    from sklearn.metrics import roc_curve, auc

    os.makedirs("results/plots", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Calibration Plot: Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/surformer_binary_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Calibration Plot: Train vs Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/surformer_binary_accuracy_curve.png")
    plt.close()

    # === Final ROC/AUC Plot ===
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, label in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(label.numpy())

    # Compute ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Surformer Binary Classification")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("results/plots/surformer_binary_roc_auc.png")
    plt.close()

end_time = time.time()
elapsed = end_time - start_time
# Print to stdout 
print(f"Total script running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
# Save to file
os.makedirs("results/plots", exist_ok=True)
with open("results/plots/surformer_binary_runtime.txt", "w") as f:
    f.write(f"Running time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n")
