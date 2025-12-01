import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling, FloatRandomSampling
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

class CustomCallback(Callback):
    
    def __init__(self) -> None:
        self.start = 0
        super().__init__()

    def initialize(self, algorithm):
        # print(f"{I} Callback initialized for geeration {algorithm.n_iter}")
        pass
    
    def notify(self, algorithm):
        # print(f"{I} Nofity for geeration {algorithm.n_iter}")
        pass


class CustomBinaryRandomSampling(Sampling):
    
    def __init__(self, val=0.2):
        self.val = val
        super().__init__()

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
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

    def _evaluate(self, x, out):
        # simulate a progress bar
        # print(f"Evaluation: {'-'*(progress-1)}{progress}", end='\r')
        ind = x
        # ind = x < 0.25
        n_genes = ind.sum()
        # print(ind)
        # print(type(ind))
        # print(n_genes)
        # print(f"Number of genes for this individual: {n_genes}")
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
POP_SIZE = 40
OFFSPRING = 10
TERMINATION_GEN = 100
DATASET_PATH = './GSE19429_Biomarker_Input.csv'
I = "[i]"

np.random.seed(SEED)

print(f"{I} Loading dataset {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

print(f"{I} processing the data...")
data = df.drop(columns=['Unnamed: 0','target']).values
# CRUCIAL step, normalize the data
data = StandardScaler().fit_transform(data)
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
clf = SVC(kernel='linear', class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"{I} Defining problem...")
problem = BiomarkerIdentification(data, labels, clf, cv)

print(f"{I} Defining algorithm...")
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=BinaryRandomSampling(),
    crossover=TwoPointCrossover(),
    mutation=BitflipMutation(),
    n_offsprings=OFFSPRING,
    callback=CustomCallback(),
    eliminate_duplicates=True)

# link to possible termination criterions https://pymoo.org/interface/termination.html
termination = get_termination("n_gen", TERMINATION_GEN)

print("[✓] Starting optimization...")
res = minimize(problem,
               algorithm,
               termination,
               seed=SEED,
               save_history=True,
               verbose=True)

