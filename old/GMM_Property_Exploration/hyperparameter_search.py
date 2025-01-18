#This program varies the hyperparameters of sklearn's Bayesian Dirichlet Gaussian Mixture Model and estimates
# the variation in its outputs for a constant 2D simulated input

import numpy as np
from sklearn import mixture

np.random.seed(1) #Set numpy random seed

class SimCat:
    def __init__(self, mean1:float, mean2:float, v1:float, v2:float, cov12:float):
        #mean1: mean on dimension 1
        #mean2: mean on dimension 2
        #v1: variance on dimension 1
        #v2: variance on dimension 2
        #cov12: covariance of dimension 1 and 2
        self.mean = [mean1,mean2]
        self.cov = [[v1, cov12], [cov12, v2]]

    def sample(self, num_samples:int):
        #Generates samples from normal distribution defined by self.mean and self.cov
        #Generates num_samples number of samples
        return np.random.default_rng().multivariate_normal(self.mean, self.cov, num_samples)

class SimDist:
    #cats: collection of categories of type Category
    def __init__(self, cats:list):
        self.cats = cats

    def sample(self, num_samples_per_cat:int):
        self.samples = np.concatenate(tuple([cat.sample(num_samples_per_cat) for cat in self.cats]))


class HyperparamSetting:
    def __init__(self,
                 num_samples: int, #Number of input samples per category
                 weight_conc_prior: float, #Affects probability of generating new categories - higher = more categories
                 max_cats:int, #Maximum number of categories to consider in approximation to Dirichlet
                 covariance_type:str = 'full', #Whether to estimate each category's covariance matrix separately, or to make them share parts
                 init_params:str = 'random_from_data', #Whether to initialize weights, means, and covariances randomly, random from data, or by running kmeans first
                 random_seed:int = 1):
        self.learner = mixture.BayesianGaussianMixture(
                                                weight_concentration_prior=weight_conc_prior,
                                                n_components=max_cats,
                                                covariance_type=covariance_type,
                                                init_params=init_params,
                                                random_state=random_seed
                                                )
        self.num_samples = num_samples

    def run(self, input_dist:SimDist, print_output=False):
        samples = input_dist.sample(self.num_samples)
        learn_output = self.learner.fit(samples)
        if print_output:
            print("Number of categories:", len(learn_output.means_))
            for index,mean in enumerate(learn_output.means_):
                print("Mean:",mean, "Cov:", learn_output.covariances_[index])

    def evaluate(self): #Implementing adjusted rand index from scratch, for pedagogy
        return

    #Rand index
    #categorizationX: dictionary from point(float) to name of category it belongs to(int)
    #categorizationY: dictionary from point(float) to name of category it belongs to(int)
    def rand(self, categorizationX:dict, categorizationY:dict): #Implementing rand index from scratch, for learning
        #Iterate over the pairs of points in categorization 1 (pair order doesn't matter)
        pairs = []
        #Get categories of points in categorizationX

        #Check if same category or different

        #Get categories of points in categorizationY

        #Check if same category or different

        #Based on same/different in X/Y, increment the right count

        pairs_same_XY = 0
        pairs_diff_XY = 0
        pairs_sameX_diffY = 0
        pairs_diffX_sameY = 0



