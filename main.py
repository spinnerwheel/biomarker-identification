import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score

import argparse

from utils.data_processing import process_data
from utils.utils import check_biomarker_validity, get_accuracy, get_number_of_genes, plot_pareto_front, print_results, save_run, load_run
from time import strftime


class CustomCallback(Callback):
    """
    Explain here what this class does
    """
    
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


class CustomBinaryRandomSampling(Sampling):
    """
    Explain here what this class does
    """


    def __init__(self, val=0.5):
        self.val = val
        super().__init__()

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        if random_state is None:
            print("random_state is None")
        val = random_state.random((n_samples, problem.n_var))
        return (val < self.val).astype(bool)

class BiomarkerIdentification(ElementwiseProblem):
    """
    Explain here what this class does
    """


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


# Define Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Biomarker Identification using NSGA-II")

    parser.add_argument("--pop_size", type=int, default=1000, help="Population size for NSGA-II")
    parser.add_argument("--offspring", type=int, default=500, help="Number of offsprings per generation")
    parser.add_argument("--termination_gen", type=int, default=100, help="Maximum number of generations before termination")
    parser.add_argument("--dataset_path", type=str, default="./datasets/GSE19429_Biomarker_Input.csv", help="Path to the input dataset CSV file")
    parser.add_argument("--dataset_name", type=str, default="GSE19429", help="Name of the dataset (for logging purposes)")
    parser.add_argument("--clf", type=str, default="SVC", help="Classifier to use (default: SVC)")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds for Stratified K-Fold CV")
    parser.add_argument("--save_history", type=bool, default=True, help="Flag to save optimization history")
    parser.add_argument("--verbose", type=bool, default=True, help="Flag to enable verbose output")
    parser.add_argument("--n_max_evals", type=int, default=100000, help="Maximum number of evaluations before termination")
    parser.add_argument("--ftol", type=float, default=0.0025, help="Function tolerance for termination")

    return parser.parse_args()


def run_experiment(seed, pop_size, offspring, termination_gen, data, labels, n_max_evals, I, ftol):
    np.random.seed(seed)
    
    print(f"{I} Defining classifier...")
    # clf = SVC(kernel='poly', class_weight='balanced', random_state=seed)
    clf = SVC(class_weight='balanced', random_state=seed)
    # ada = AdaBoostClassifier(estimator=clf, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print(f"{I} Defining problem...")
    problem = BiomarkerIdentification(data, labels, clf, cv)

    print(f"{I} Defining algorithm...")
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=CustomBinaryRandomSampling(val=0.02),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        n_offsprings=offspring,
        callback=CustomCallback(),
        save_history=False,
        eliminate_duplicates=True)

    # link to possible termination criterions https://pymoo.org/interface/termination.html
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=ftol,
        period=20,
        n_max_gen=termination_gen,
        n_max_evals=n_max_evals,
    )

    print("[V] Starting optimization...")
    res = minimize(problem,
                algorithm,
                termination,
                seed=seed,
                verbose=True)

    return res, problem


def main():

    args = parse_args()

    SEED = 424242
    POP_SIZE = args.pop_size
    OFFSPRING = args.offspring
    TERMINATION_GEN = args.termination_gen
    # DATASET_PATH = args.dataset_path
    DATASET_NAME = args.dataset_name
    I = "[i]"

    print(f"{I} Biomarker Identification using NSGA-II")
    print(f"{I} Population Size: {POP_SIZE}")
    print(f"{I} Offspring per Generation: {OFFSPRING}")
    print(f"{I} Termination Generation: {TERMINATION_GEN}")

    # Load Data
    data, labels, gene_names = process_data(DATASET_NAME, I)

    # Run Experiment
    res, problem = run_experiment(SEED, POP_SIZE, OFFSPRING, TERMINATION_GEN, data, labels, 100000, I, args.ftol)

    save_run(res)

    # TODO
    # Pareto front plot
    # Average, min, max genes for generation
    # Distribution of genes

    # Get Results
    # Convert Error back to Accuracy
    accuracy = get_accuracy(res)
    # Number of Genes Selected
    num_genes = get_number_of_genes(res, problem)

    # Sort
    sorted_idx = np.argsort(accuracy)
    accuracy = accuracy[sorted_idx]
    num_genes = num_genes[sorted_idx]
    X_res = res.X[sorted_idx]

    # Plot Pareto Front
    plot_pareto_front(DATASET_NAME, num_genes, accuracy)

    # Print Results
    print_results(accuracy, num_genes, X_res, gene_names)

    # Y-Scrambled validation
    choice = input("Do you want to run Y-Scrambled validation? [Y/n] > ")
    if (choice.strip().upper() not in ["", "Y"]):
        print("Done.")
        exit(1)

    print(f"{I} Running Y-Scrambled to validate the results...")

    # # Redefine all the elements with the new SEED
    SEED = 696969
    np.random.seed(SEED)

    # Shuffle Labels
    labels_shuffled = np.random.permutation(labels)

    # Run Experiment with Scrambled Labels
    res_shuffled, _ = run_experiment(SEED, POP_SIZE, OFFSPRING, TERMINATION_GEN, data, labels_shuffled, 100000, I, args.ftol)

    # Get Best Scrambled Accuracy
    shuffled_acc = get_accuracy(res_shuffled)

    # Check Validity of Biomarkers
    check_biomarker_validity(accuracy, shuffled_acc)


if __name__ == "__main__":
    main()
