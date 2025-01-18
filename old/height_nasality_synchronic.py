import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from old.explore_simulations import plot_results
np.random.seed(1) #Set numpy's random seed



class LanguageInput:
    '''
    Each LanguageInput object represents an individual's
    system of Gaussian vowel categories defined on height and nasality dimensions.
    A LanguageInput consists of a list of Category objects, self.vowels, and useful methods
    on that list of Category objects:
    sampling from them (sample),
    running learning on a noisy sample of them (simulate_learning)
    helper function for setting up initial distribution (starter_distribution)
    '''
    def __init__(self):
        '''
        Initializes LanguageInput with no vowel categories
        '''
        self.vowels = [] #List of Category objects
        return

    def sample(self, num_samples_per_cat):
        '''
        Samples num_samples_per_cat vowels per vowel category in self.vowels
        :param num_samples_per_cat: int, number of samples to generate of each vowel category
        :return: numpy array of shape num_samples_per_cat * number of categories x 2
        '''
        return np.concatenate(tuple([vowel.sample(num_samples_per_cat) for vowel in self.vowels]))

    def simulate_learning(self, weight_conc_prior = None, num_samples=500):
        '''
        Runs a Dirichlet Mixture of Gaussians learner on the vowel distribution defined by self.vowels
        and plots the learned categories
        :param num_samples Number of random samples to take from each Gaussian category
        :param weight_conc_prior: Concentration prior for the Dirichlet process Gaussian Mixture learners
        :return: None
        '''
        samples = self.sample(num_samples)
        # Fit a Dirichlet Gaussian mixture with max 5 components
        dpgmm = mixture.BayesianGaussianMixture(n_components=10, covariance_type="full",
                                                 weight_concentration_prior=weight_conc_prior).fit(samples)
        plot_results(
            samples,
            dpgmm.predict(samples),
            dpgmm.means_,
            dpgmm.covariances_,
            0,
            "Bayesian Gaussian Mixture with a Dirichlet process prior",
            xmin=50,
            xmax=250,
            ymin=50,
            ymax=350
        )
        plt.show()



#Each Category object represents a vowel category defined by a mean nasality, height value
#and a covariance matrix (cov)
class Category:
    def __init__(self, mean_nasality, mean_height, s_nasality, s_height, c_nasality_height):
        '''
        Defines this.mean (1d list): this.mean[0] = mean_nasality, this.mean[1] = mean_height
        Defines this.cov (2d list): covariance matrix for height and nasality
        :param mean_nasality: float,mean of this category on nasality dimension
        :param mean_height: float, mean of this category on height dimension
        :param s_nasality: float,variance on nasality dimension
        :param s_height: float,variance on height dimension
        :param c_nasality_height: float, covariance of nasality and height
        '''
        self.mean = [mean_nasality, mean_height]
        self.cov = [[s_nasality, c_nasality_height],[c_nasality_height, s_height]]

    # def sample(self, num_samples):
    #     '''Returns num_samples (int) number of random samples from the distribution defined by
    #     self.mean and self.cov
    #     Returned random samples are a num_samples x 2 numpy array'''
    #     print(self.cov)
    #     return np.random.default_rng().multivariate_normal(self.mean, self.cov, num_samples)



#Set up underlying distributions for a language
language1 = LanguageInput()
nasality_height_corr = 30
high = Category(mean_nasality=100, mean_height=250,s_nasality=80,s_height=80,c_nasality_height=nasality_height_corr)
med = Category(mean_nasality=110, mean_height=200, s_nasality=80, s_height=80, c_nasality_height=nasality_height_corr)
low = Category(mean_nasality=120, mean_height=150, s_nasality=80, s_height=80, c_nasality_height=nasality_height_corr)

#Shifting by varying amounts from oral - not quite a Gaussian because variance not symmetrically increased
high_coart_n = Category(mean_nasality=110, mean_height=240,s_nasality=120,s_height=80,c_nasality_height=nasality_height_corr)

high_all_n = Category(mean_nasality=120, mean_height=240,s_nasality=80,s_height=80,c_nasality_height=nasality_height_corr)
med_n = Category(mean_nasality=120, mean_height=200, s_nasality=80, s_height=80, c_nasality_height=nasality_height_corr)
low_n = Category(mean_nasality=120, mean_height=160, s_nasality=80, s_height=80, c_nasality_height=nasality_height_corr)
#Is nasal leakage bigger for low vowels?

language1.vowels += [high, med, low, high_coart_n, med_n, low_n]
language2 = LanguageInput()
language2.vowels += [high, low, high_coart_n, low_n]



#Run learning simulations
language1.simulate_learning()#weight_conc_prior=1 * 10 ** -10)
language2.simulate_learning()#weight_conc_prior=1 * 10 ** -10)




