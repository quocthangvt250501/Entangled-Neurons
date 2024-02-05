import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.decomposition import PCA

import os
import math

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Classical_NeuralNetwork:

    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        self.pca = None
        self.X_pca = None
        self.cm = None
        self.classes = ['Weak Binder', 'Strong Binder']
        
    
    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):

        cm = self.cm
        classes = self.classes

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

    
    def plot_pca_variance(self):
        
        pca = self.pca

        plt.figure(figsize=(10, 6))
        plt.title('Variance of the Principal Components')
        plt.plot(pca.explained_variance_)
        plt.xlabel('ith Principal Component')
        plt.ylabel('Variance')
        plt.yscale('log')
        plt.show()

    
    def plot_pca_2D(self):

        X_pca = self.X_pca

        plt.figure(figsize=(10, 6))
        plt.title('Labeled samples in the latent space of the two first PC')
        plt.scatter(X_pca[:,0], X_pca[:,1], c= self.y)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()



    def train(self):
        
        X = self.X
        y = self.y

        pca = PCA(n_components=10) 
        X_pca = pca.fit_transform(X)

        self.pca = pca
        self.X_pca = X_pca

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        accuracy_scores = []
        confusion_matrices = []
        fold_accuracies = []

        for train_index, test_index in kf.split(X_pca): 
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = MLPClassifier(solver='adam', hidden_layer_sizes=(10,), max_iter=1, warm_start=True, learning_rate='adaptive', alpha=0.0001, shuffle=True, random_state=42, activation='logistic') #activation testet: 'relu', 'tanh' and 'logistic'.
            
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
        self.cm = np.mean(confusion_matrices, axis=0).astype(int)

        print("Average AUC:", np.mean(auc_scores))
        print("Average Accuracy:", np.mean(accuracy_scores))