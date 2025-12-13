import numpy as np
import matplotlib.pyplot as plt
from os import path
import bz2
import pickle

def get_accuracy(res):
    return 1.0 - res.F[:, 0]


def get_number_of_genes(res, problem):
    return (res.F[:, 1] * problem.n_var).astype(int)


def plot_pareto_front(dataset_name, num_genes, accuracy):
    # Plot Pareto Front
    plt.figure(figsize=(10, 6))
    plt.scatter(num_genes, accuracy, s=80, c='blue', edgecolors='k')
    plt.plot(num_genes, accuracy, linestyle='--', color='gray', alpha=0.5)
    plt.title(f"Pareto Front ({dataset_name})\nOptimizing Balanced Accuracy for Imbalanced Data")
    plt.xlabel("Number of Genes")
    plt.ylabel("Balanced Accuracy (5-Fold CV)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"Pareto_{dataset_name}.png")
    plt.show()


def print_results(accuracy, num_genes, X_res, gene_names):
    # Print Best Solutions
    print("\n" + "="*50)
    print(" PARETO OPTIMAL SOLUTIONS (Sorted by number of genes)")
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


def check_biomarker_validity(accuracy, shuffled_acc):
    best_shuffled = np.max(shuffled_acc)
    best_real = np.max(accuracy)

    print(f"Real Balanced Accuracy:      {best_real:.4f}")
    print(f"Scrambled Balanced Accuracy: {best_shuffled:.4f}")

    if best_shuffled < (best_real - 0.15):
        print("\nCONCLUSION: PASS. The biomarkers are valid signal.")
    else:
        print("\nCONCLUSION: WARNING. The model might be overfitting.")

        
def save_run(res, filename=None):
    """
    Save results of a run serialized in a file.
    """
    if filename is None:
        filename = "runs/" + strftime("%j_%H:%M:%S.run")
    with bz2.open(filename, "wb") as f:
        pickle.dump(res, f)
    

def load_run(filename: str):
    """
    Return results of a previous run, loaded from file.
    """
    res = None
    if path.exists(filename) and path.isfile(filename):
        with bz2.open(filename, "rb") as f:
            res = pickle.load(f)
    return res
