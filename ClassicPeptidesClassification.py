import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.decomposition import PCA

try:
    import epitopepredict as ep
except ImportError:
    print("epitopepredict is not installed. Please install or check your PYTHONPATH.")
    sys.exit(1)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

blosum = ep.blosum62

def blosum_encode(seq):
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    return x.values.flatten()

def convert_ic50_to_binary(ic50_values, threshold=500):
    return np.where(ic50_values <= threshold, 1, 0)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def train_predictor_with_cv_and_cm(allele, encoder):
    data = ep.get_training_set(allele, length=9)[:100]
    data['binary_label'] = convert_ic50_to_binary(data['ic50'])
    X = np.array(data['peptide'].apply(encoder).tolist())
    y = data['binary_label'].values
    
    pca = PCA(n_components=10) 
    X_pca = pca.fit_transform(X)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    accuracy_scores = []
    confusion_matrices = []
    fold_accuracies = []

    for train_index, test_index in kf.split(X_pca): 
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(50,), max_iter=1, warm_start=True, learning_rate='adaptive', alpha=0.0001, shuffle=True, random_state=42, activation='logistic') #activation testet: 'relu', 'tanh' and 'logistic'.
        
        n_iterations = 100
        iteration_accuracies = []
        
        for _ in range(n_iterations):
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            iteration_accuracy = accuracy_score(y_test, predictions)
            iteration_accuracies.append(iteration_accuracy)
        
        fold_accuracies.append(iteration_accuracies)
        
        proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        accuracy = accuracy_score(y_test, predictions)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        cm = confusion_matrix(y_test, predictions)
        confusion_matrices.append(cm)

    # Plot AUC and Accuracy Scores per Fold
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, kf.get_n_splits() + 1), auc_scores, label='AUC Scores', marker='o')
    plt.plot(range(1, kf.get_n_splits() + 1), accuracy_scores, label='Accuracy Scores', marker='x')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('AUC and Accuracy Scores per Fold')
    plt.xticks(range(1, kf.get_n_splits() + 1))
    plt.legend()
    plt.show()

    # Plot Average Validation Accuracy vs. Iteration across all folds
    avg_accuracies = np.mean(fold_accuracies, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_iterations + 1), avg_accuracies, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Average Validation Accuracy vs. Iteration across all folds')
    plt.legend()
    plt.show()

    # Average confusion matrix over all folds
    mean_cm = np.mean(confusion_matrices, axis=0).astype(int)
    plt.figure()
    plot_confusion_matrix(mean_cm, classes=['Weak Binder', 'Strong Binder'], title='Average Confusion Matrix')
    plt.show()

    print("Average AUC:", np.mean(auc_scores))
    print("Average Accuracy:", np.mean(accuracy_scores))

allele = 'HLA-A*03:01'
train_predictor_with_cv_and_cm(allele, blosum_encode)