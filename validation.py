import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, classification_report

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_PATH = 'GA_Input_Data_GSE19429_Symbols.csv'
TEST_PATH  = 'GA_Input_Data_GSE58831.csv'
GENE_LIST  = 'best_genes.txt'
SEED = 42

# ==============================================================================
# 1. LOAD GENE PANEL
# ==============================================================================
print(f"[i] Loading Gene Panel from {GENE_LIST}...")
try:
    with open(GENE_LIST, "r") as f:
        original_panel = [line.strip() for line in f if line.strip()]
    print(f"    - Original Panel ({len(original_panel)} genes): {original_panel}")
except FileNotFoundError:
    print(f"[!] Error: Could not find '{GENE_LIST}'.")
    exit()

# ==============================================================================
# 2. DATA LOADING (INTERSECTION STRATEGY)
# ==============================================================================
print(f"\n[i] Loading Datasets...")
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

# Identify Target Columns
col_train = next((c for c in ['Class_Label', 'target', 'Group'] if c in df_train.columns), df_train.columns[-1])
col_test  = next((c for c in ['Class_Label', 'target', 'Group'] if c in df_test.columns), df_test.columns[-1])

# --- FIND COMMON GENES ---
# We only use genes that exist in BOTH the panel AND the test set
available_in_train = set(df_train.columns)
available_in_test  = set(df_test.columns)
panel_set = set(original_panel)

# The "Intersection" -> Genes present in Panel AND Train AND Test
final_genes = list(panel_set.intersection(available_in_train).intersection(available_in_test))

print(f"\n[i] Gene Intersection Check:")
print(f"    - Missing in Test Set: {panel_set - available_in_test}")
print(f"    - FINAL GENES USED:    {len(final_genes)} genes -> {final_genes}")

if len(final_genes) == 0:
    print("[!] Error: No common genes found. Validation cannot proceed.")
    exit()

# Extract Data using ONLY the common genes
X_train = df_train[final_genes].values
y_train_raw = df_train[col_train]

X_test = df_test[final_genes].values
y_test_raw = df_test[col_test]

# ==============================================================================
# 3. ENCODE LABELS
# ==============================================================================
# Helper to map "MDS" -> 1 and "Control" -> 0 safely
def encode_mds(y_data):
    y_enc = []
    for label in y_data:
        s = str(label).upper()
        if "MDS" in s or "DISEASE" in s:
            y_enc.append(1)
        else:
            y_enc.append(0)
    return np.array(y_enc)

y_train = encode_mds(y_train_raw)
y_test  = encode_mds(y_test_raw)

print(f"\n[i] Final Data Dimensions:")
print(f"    - Train (GSE4619):  {X_train.shape}")
print(f"    - Test  (GSE19429): {X_test.shape}")

# ==============================================================================
# 4. TRAINING & PREDICTION (WITH BATCH EFFECT CORRECTION)
# ==============================================================================
print(f"\n[i] Applying Independent Scaling to remove Batch Effects...")

# We fit a scaler on Train AND fit a separate scaler on Test.
# This aligns the center of both datasets to 0, removing the baseline shift.

scaler_train = RobustScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

scaler_test = RobustScaler()
X_test_scaled  = scaler_test.fit_transform(X_test) # Fit on X_test, not X_train

print(f"[i] Training SVM on GSE4619...")
clf = SVC(class_weight='balanced', kernel='rbf', random_state=SEED, verbose = True)
clf.fit(X_train_scaled, y_train)

print(f"[i] Predicting on GSE19429...")
y_pred = clf.predict(X_test_scaled)

# ==============================================================================
# 5. RESULTSc
# ==============================================================================
bal_acc = balanced_accuracy_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print(f" VALIDATION RESULTS: GSE19429")
print("="*50)
print(f"Balanced Accuracy:  {bal_acc:.4f}")
print(f"Raw Accuracy:       {acc:.4f}")
print("-" * 50)
print("Confusion Matrix:")
print(conf_mat)
print("-" * 50)

# Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Control', 'MDS'], yticklabels=['Control', 'MDS'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'GSE19429 Validation\nGenes: {final_genes}\nBalanced Acc: {bal_acc:.2f}')
plt.tight_layout()
plt.savefig("Validation_GSE19429.png")
print("[✓] Saved plot to 'Validation_GSE19429.png'")
plt.show()

# Interpretation
if bal_acc > 0.70:
    print("\n[SUCCESS] The reduced panel validates successfully!")
else:
    print("\n[FAIL] The markers did not generalize to the new dataset.")