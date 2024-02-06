import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold

from qiskit.algorithms.optimizers import ADAM, SPSA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import BackendSampler
from qiskit_algorithms.gradients import ParamShiftSamplerGradient

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss


class Quantum_NeuralNetwork:

    def __init__(self, X_pca, y, backend):
        
        self.y = y
        self.X_pca = X_pca
        self.cm = None
        self.classes = ['Weak Binder', 'Strong Binder']
        self.spsa_loss_recorder = []
        self.backend = backend
        
    
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
    
    def get_OHD(self, y):
        """
        Only works for 2D encoding
        """
        return np.array([y_[1] for y_ in y])
    
    def one_qubit_binary(self, x):
        return x % 2


    def spsa_callback(self, nfev, params, fval, stepsize, accepted=None):
        if (nfev % 3) == 0:
            self.spsa_loss_recorder.append(fval)

    def build_network(self,
                    num_classes = 2,
                    num_features = 2,
                    reps = 3):


        # Data embedding
        fmap_zz = ZZFeatureMap(feature_dimension = num_features, 
                               reps = reps, 
                               entanglement = 'linear')
        input_params = fmap_zz.parameters
        
        # Anstaz
        ansatz_tl = TwoLocal(num_qubits = num_features, 
                             rotation_blocks = ['ry', 'rz'], 
                             entanglement_blocks= 'cx' , 
                             entanglement = 'full', #'linear'
                             reps = reps)
        
        weights_params = ansatz_tl.parameters

        # Assign random weights
        weights = np.random.random(len(ansatz_tl.parameters))
        ansatz_tl.assign_parameters({k:v for (k,v) in zip(ansatz_tl.parameters, weights)})

        var_circuit = fmap_zz.compose(ansatz_tl)

        # Primitives
        options = {}
        sampler = BackendSampler(backend=self.backend, options=options)

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

        return len(ansatz_tl.parameters), sampler_qnn, var_circuit_with_meas



    def train(self,
              n_splits = 2,
              n_iterations = 10,
              num_features = 10,
              reps = 2):

        X_pca = self.X_pca
        y = self.y

        kf = KFold(n_splits = n_splits, 
                   shuffle = True, 
                   random_state = 42)
    

        for index_fold , (train_index, test_index) in enumerate(kf.split(X_pca)): 

            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y[train_index], y[test_index]

            len_init , sampler_qnn, var_circuit_with_meas = self.build_network(num_classes = 2,
                                                        num_features = num_features,
                                                        reps = reps)
            
            y_train_1h = self.get_OHE(y_train)
            y_test_1h = self.get_OHE(y_test)

            print(f"Label {y_train[1]} converted to {y_train_1h[1]}")
            print(f"Label {y_train[0]} converted to {y_train_1h[0]}")


            spsa_opt = SPSA(maxiter = n_iterations, 
                            callback = self.spsa_callback)
            
            #self.spsa_loss_recorder = []
            accuracies = np.zeros((2, n_splits, n_iterations))

            initial_point = np.random.random((len_init,))
            vqc = NeuralNetworkClassifier(neural_network = sampler_qnn,
                                        loss = CrossEntropyLoss(),
                                        one_hot = True,
                                        optimizer = spsa_opt,
                                        initial_point = initial_point)

            #for index_itr in tqdm(range(n_iterations), total = n_iterations):
            vqc.fit(X_train, y_train_1h)

            print("Train Accuracy:", vqc.score(X = X_train, y = y_train_1h))
            print("Test Accuracy:", vqc.score(X = X_test, y = y_test_1h))

            
            plt.figure()
            plt.plot(self.spsa_loss_recorder)
            plt.xlabel("Number of epochs")
            plt.ylabel("Cross Entropy Loss")
            plt.title("Training loss")
            plt.show()

            self.spsa_loss_recorder = []

            #return var_circuit_with_meas.draw('mpl', scale=0.7)
   
