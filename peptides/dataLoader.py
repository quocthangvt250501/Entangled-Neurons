import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
import math
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class dataLoader:

    def __init__(self):

        self.path_dataset = 'dataset/'
        self.allele = 'HLA-A*03:01'
        self.blosum = pd.read_csv(os.path.join(self.path_dataset,'blosum62.csv'))
        self.data = None
        self.X_pca = None
        self.pca = None
        self.y = None


    def aff2log50k(self, a):
        """
        From package : epitopepredict
        """
        return 1 - (math.log(a) / math.log(50000))
    

    def convert_ic50_to_binary(self,
                               ic50_values, 
                               threshold=500):
        """
        From package : epitopepredict
        """
        return np.where(ic50_values <= threshold, 1, 0)
    

    def blosum_encode(self,
                      seq):
        """   
        From package : epitopepredict
        """
        x = pd.DataFrame([self.blosum[i] for i in seq]).reset_index(drop=True)
        return x.values.flatten()
        

    def get_training_set(self,
                         allele=None, 
                         length=None):
        """
        Get training set for MHC-I data.
        
        From package : epitopepredict
        """

        b = pd.read_csv(os.path.join(self.path_dataset, 'curated_training_data.no_mass_spec.zip'))
        eval1 = self.get_evaluation_set1()
        df = b.loc[~b.peptide.isin(eval1.peptide)].copy()
        if allele is not None:
            df = b.loc[b.allele==allele].copy()

        df['log50k'] = df.ic50.apply(lambda x: self.aff2log50k(x))
        df['length'] = df.peptide.str.len()
        if length != None:
            df = df[(df.length==length)]
        df = df[df.ic50<50000]
        df = df[df.measurement_type=='quantitative']
        return df
    
    def get_evaluation_set1(self,
                            allele=None, 
                            length=None):
        """
        Get eval set of peptides
        
        From package : epitopepredict
        """
        e = pd.read_csv(os.path.join(self.path_dataset, 'binding_data_2013.zip'),comment='#')
        
        if allele is not None:
            e = e[e.allele==allele]
        if length != None:
            e = e[(e.length==length) ]
        e['log50k'] = e.ic50.apply(lambda x: self.aff2log50k(x)).round(2)
        return e
    
    
    def load_data(self, allele):

        data = self.get_training_set(allele, length=9)[:100]
        data['binary_label'] = self.convert_ic50_to_binary(data['ic50'])
        encoder = self.blosum_encode

        X = np.array(data['peptide'].apply(encoder).tolist())
        self.y = data['binary_label'].values

        self.pca = PCA(n_components=10) 
        self.X_pca = self.pca.fit_transform(X)

        return self.X_pca, self.y


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