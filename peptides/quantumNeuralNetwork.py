import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold

from qiskit.algorithms.optimizers import ADAM, SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.gradients import ParamShiftSamplerGradient

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss


class Quantum_NeuralNetwork:

    def __init__(self, X_pca, y):
        
        self.y = y
        self.X_pca = X_pca
        self.cm = None
        self.classes = ['Weak Binder', 'Strong Binder']
        self.spsa_loss_recorder = []
        
    
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

    def get_OHE(self, y):
        unique_labels = np.unique(y, axis=0)
        y_one_hot = [(np.eye(len(unique_labels), dtype=int)[np.where(unique_labels == y_i)]).reshape(len(unique_labels)) for y_i in y]
        return np.array(y_one_hot)
    
    def one_qubit_binary(self, x):
        return x % 2


    def spsa_callback(self, nfev, params, fval, stepsize, accepted=None):
        """
        Can be used for SPSA and GradientDescent optimizers
        nfev: the number of function evals
        params: the current parameters
        fval: the current function value
        stepsize: size of the update step
        accepted: whether the step was accepted (not used for )
        """

        if (nfev % 3) == 0:
            self.spsa_loss_recorder.append(fval)

    def build_network(self,
                    num_classes = 2,
                    num_features = 2):
        
        # Initialization
        X_pca = self.X_pca
        SEED = 123
        qasm_sim = AerSimulator()

        # Data embedding
        fmap_zz = ZZFeatureMap(feature_dimension = num_features, 
                               reps = 2, 
                               entanglement = 'linear')
        input_params = fmap_zz.parameters
        #fmap_zz.assign_parameters({k:v for (k,v) in zip(fmap_zz.parameters, X_pca[0])}).decompose().draw('mpl', scale=0.7)
        
        # Anstaz
        ansatz_tl = TwoLocal(num_qubits = num_features, 
                             rotation_blocks = ['ry', 'rz'], 
                             entanglement_blocks= 'cx' , 
                             entanglement = 'linear', 
                             reps = 2)
        weights_params = ansatz_tl.parameters
        #ansatz_tl.decompose().draw('mpl', scale=0.7)

        # Assign random weights
        weights = np.random.random(len(ansatz_tl.parameters))
        ansatz_tl.assign_parameters({k:v for (k,v) in zip(ansatz_tl.parameters, weights)})

        var_circuit = fmap_zz.compose(ansatz_tl)
        #var_circuit.draw('mpl', scale=0.7)

        # Primitives
        options = {}
        sampler = BackendSampler(backend=qasm_sim, options=options)
        estimator = BackendEstimator(backend=qasm_sim, options=options)

        # Circuit with measure
        var_circuit_with_meas = var_circuit.copy()
        var_circuit_with_meas.measure_all()

        paramShiftSampGrad = ParamShiftSamplerGradient(sampler=sampler)
        sampler_qnn = SamplerQNN(sampler = sampler,
                                circuit = var_circuit_with_meas,
                                input_params = input_params,   
                                weight_params = weights_params,
                                interpret = self.one_qubit_binary,
                                output_shape = 2,
                                gradient = paramShiftSampGrad)

        # Forward pass
        output = sampler_qnn.forward(X_pca, weights)

        # Backward pass
        _, weights_grad = sampler_qnn.backward(X_pca, weights)

        return len(ansatz_tl.parameters), sampler_qnn



    def train(self):

        X_pca = self.X_pca
        y = self.y

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        auc_scores = []
        accuracy_scores = []
        confusion_matrices = []
        fold_accuracies = []

        for train_index, test_index in kf.split(X_pca): 

            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y[train_index], y[test_index]

            len_init , sampler_qnn = self.build_network(num_classes = 2,
                                                        num_features = 3)
            
            y_train_1h = self.get_OHE(y_train)
            y_test_1h = self.get_OHE(y_test)

            print(f"Label {y_train[1]} converted to {y_train_1h[1]}")
            print(f"Label {y_train[0]} converted to {y_train_1h[0]}")

            max_itr = 50
            spsa_opt = SPSA(maxiter = max_itr, 
                            callback = self.spsa_callback)
            self.spsa_loss_recorder = []

            initial_point = np.random.random((len_init,))
            vqc = NeuralNetworkClassifier(neural_network = sampler_qnn,
                                        loss = CrossEntropyLoss(),
                                        one_hot = True,
                                        optimizer = spsa_opt,
                                        initial_point = initial_point)

            vqc = vqc.fit(X_train, y_train_1h)

            plt.figure()
            plt.plot(self.spsa_loss_recorder)
            plt.xlabel("Number of epochs")
            plt.title("Training loss")
            plt.show()


            score_train = vqc.score(X_train, y_train_1h)
            score_test = vqc.score(X_test, y_test_1h)
            print(f'Score on the train set {score_train}')
            print(f'Score on the test set {score_test}')


            y_train_pred = vqc.predict(X_train)
            y_test_pred = vqc.predict(X_test)

            train_pred_acc = np.all(y_train_pred == y_train_1h, axis=1)
            test_pred_acc = np.all(y_test_pred == y_test_1h, axis=1)


            x_all = np.concatenate((X_train, X_test))
            y_all= np.concatenate((y_train_1h, y_test_1h))
            y_pred_acc_all = np.concatenate((train_pred_acc, test_pred_acc))

            x_b = x_all[np.all(y_all == [1, 0], axis=1)]
            x_b_good = x_b[(y_pred_acc_all[np.all(y_all == [1, 0], axis=1)])]
            x_b_bad = x_b[np.logical_not(y_pred_acc_all[np.all(y_all == [1, 0], axis=1)])]

            x_r = x_all[np.all(y_all == [0, 1], axis=1)]
            x_r_good = x_r[(y_pred_acc_all[np.all(y_all == [0, 1], axis=1)])]
            x_r_bad = x_r[np.logical_not(y_pred_acc_all[np.all(y_all == [0, 1], axis=1)])]


            plt.figure()
            plt.scatter(x_b_good[:,0], x_b_good[:,1], c='b', marker=".", label="Good class 0")
            plt.scatter(x_b_bad[:,0], x_b_bad[:,1], c='b', marker="x", label="Bad class 0")
            plt.scatter(x_r_good[:,0], x_r_good[:,1], c='r', marker=".", label="Good class 1")
            plt.scatter(x_r_bad[:,0], x_r_bad[:,1], c='r', marker="x", label="Bad class 1")

            plt.legend()
            plt.show()
        