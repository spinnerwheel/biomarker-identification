import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling, FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

class CustomCallback(Callback):
    
    def __init__(self) -> None:
        self.start = 0
        self.conv = []
        super().__init__()

    def initialize(self, algorithm):
        # print(f"{I} Callback initialized for geeration {algorithm.n_iter}")
        pass
    
    def notify(self, algorithm):
        # print(f"{I} Nofity for geeration {algorithm.n_iter}")
        # Get the best accuracy from the current population to measure convergence
        self.conv.append(algorithm.pop.get("F")[:,0].min())
        pass


class CustomBinaryRandomSampling(Sampling):
    
    def __init__(self, val=0.5):
        self.val = val
        super().__init__()

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        if random_state is None:
            print("random_state is None")
        val = random_state.random((n_samples, problem.n_var))
        return (val < self.val).astype(bool)

class BiomarkerIdentification(ElementwiseProblem):
    def __init__(self, X, y, clf, cv) -> None:
        self.X = X
        self.y = y
        self.clf = clf
        self.cv = cv
        super().__init__(n_var=X.shape[1],
                         n_obj=2)

    def _evaluate(self, ind, out):
        n_genes = ind.sum()
        if n_genes == 0:
            out["F"] = [1.0, 1.0]
            return

        bal_acc_scores = []
        X_sub = self.X[:, ind]

        # print(f"{I} Start training")
        for train_idx, test_idx in self.cv.split(X_sub, self.y):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            
            # Use Balanced Accuracy (Avg of Recall for both classes)
            score = balanced_accuracy_score(y_test, y_pred)
            bal_acc_scores.append(score)
        
        # print(f"{I} End of training")
        mean_bal_acc = np.mean(bal_acc_scores)
        
        # Objectives (Minimize these)
        f1 = 1.0 - mean_bal_acc     # 1 minus Balanced Accuracy
        f2 = n_genes / self.n_var   # Fraction of genes used
        
        out["F"] = [f1, f2]

SEED = 424242
POP_SIZE = 1000
OFFSPRING = 500
TERMINATION_GEN = 100
DATASET_PATH = './GSE19429_Biomarker_Input.csv'
I = "[i]"

np.random.seed(SEED)

print(f"{I} Loading dataset {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

print(f"{I} Processing the data...")
gene_names = df.drop(columns=['Unnamed: 0','target']).columns

data = df.drop(columns=['Unnamed: 0','target']).values
# CRUCIAL step, normalize the data
data = RobustScaler().fit_transform(data)
# Should avoid data copy according to
# https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
data = np.ascontiguousarray(data)

target = df['target']

print(f"Data: {data.shape}")
print("Labels: ")
unique, counts = np.unique(target, return_counts=True)
for u,c in zip(unique, counts):
    print(f"\t{u}: {c} samples")

# label MDS = 1, Healthy = 0
labels = LabelEncoder().fit_transform(target)

print(f"{I} Defining classifier...")
# clf = SVC(kernel='poly', class_weight='balanced', random_state=SEED)
clf = SVC(class_weight='balanced', random_state=SEED)
# ada = AdaBoostClassifier(estimator=clf, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"{I} Defining problem...")
problem = BiomarkerIdentification(data, labels, clf, cv)

print(f"{I} Defining algorithm...")
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=CustomBinaryRandomSampling(val=0.02),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    n_offsprings=OFFSPRING,
    callback=CustomCallback(),
    save_history=False,
    eliminate_duplicates=True)

# link to possible termination criterions https://pymoo.org/interface/termination.html
termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=20,
    n_max_gen=TERMINATION_GEN,
    n_max_evals=100000,
)

print("[✓] Starting optimization...")
res = minimize(problem,
               algorithm,
               termination,
               seed=SEED,
               verbose=True)
# TODO
# Pareto front plot
# Average, min, max genes for generation
# Distribution of genes

# Gemini stuff

F = res.F
X_res = res.X

# Convert Error back to Accuracy
accuracy = 1.0 - F[:, 0]
num_genes = (F[:, 1] * problem.n_var).astype(int)

# Sort
sorted_idx = np.argsort(accuracy)
accuracy = accuracy[sorted_idx]
num_genes = num_genes[sorted_idx]
X_res = X_res[sorted_idx]

# Plot Pareto Front
plt.figure(figsize=(10, 6))
plt.scatter(num_genes, accuracy, s=80, c='blue', edgecolors='k')
plt.plot(num_genes, accuracy, linestyle='--', color='gray', alpha=0.5)
plt.title(f"Pareto Front (GSE19429)\nOptimizing Balanced Accuracy for Imbalanced Data")
plt.xlabel("Number of Genes")
plt.ylabel("Balanced Accuracy (5-Fold CV)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("Pareto_GSE19429.png")
plt.show()

# Print Best Solutions
print("\n" + "="*50)
print(f" PARETO OPTIMAL SOLUTIONS (Sorted by number of genes)")
print("="*50)
print(f"{'# Genes':<10} | {'Bal Acc':<10} | {'Genes Selected'}")
print("-" * 50)

for i in range(len(accuracy)):
    mask = np.round(X_res[i]).astype(bool)
    selected = list(gene_names[mask])
    # Only print unique accuracy points to avoid clutter
    if i > 0 and accuracy[i] == accuracy[i-1] and num_genes[i] == num_genes[i-1]:
        continue
    print(f"{num_genes[i]:<10} | {accuracy[i]:.4f}     | {selected}")

# Y-Scrambled validation
choice = input("Do you want to run Y-Scrambled validation? [Y/n] > ")
if (choice.strip().upper() not in ["", "Y"]):
    print("Done.")
    exit(1)

print(f"{I} Running Y-Scrambled to validate the results...")

SEED = 696969

# Shuffle Labels
np.random.seed(SEED)
labels_shuffled = np.random.permutation(labels)

# Redefine all the elements with the new SEED
clf = SVC(class_weight='balanced', random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=CustomBinaryRandomSampling(val=0.02),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    n_offsprings=OFFSPRING,
    callback=CustomCallback(),
    eliminate_duplicates=True)
# Run GA on Garbage Data
problem_shuffled = BiomarkerIdentification(data, labels_shuffled, clf, cv)
res_shuffled = minimize(
    problem_shuffled,
    algorithm,
    termination,
    seed=SEED,
    verbose=True)

# Get Best Scrambled Accuracy
shuffled_acc = 1.0 - res_shuffled.F[:, 0]
best_shuffled = np.max(shuffled_acc)
best_real = np.max(accuracy)

print(f"Real Balanced Accuracy:      {best_real:.4f}")
print(f"Scrambled Balanced Accuracy: {best_shuffled:.4f}")

if best_shuffled < (best_real - 0.15):
    print("\nCONCLUSION: PASS. The biomarkers are valid signal.")
else:
    print("\nCONCLUSION: WARNING. The model might be overfitting.")
