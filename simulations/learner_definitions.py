# Program defining the Language, Learner, Category classes and the ARI and plot methods
# General structures used for running vowel inventory evolution simulations
import math

import numpy as np
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
from scipy.stats import multivariate_normal

np.random.seed(1)  # Set numpy's random seed

debug = False


class Category:
    """
    Each Category object represents a vowel category defined by a mean nasality,
    height value and a covariance matrix (cov)
    It also has a sample method, which returns a random sample from a 2D Gaussian distribution
    centered on the category's mean nasality, height with the category's covariance matrix
    """

    def __init__(self, mean, cov):
        """
        Defines this.mean and this.cov
        :param mean: Should be 1x2 (list) of mean nasality, mean_height
        :param cov: Should be 2x2 covariance matrix (list of lists) for nasality and height
        """
        self.mean = mean
        self.cov = cov

    @classmethod
    def build_params(cls, mean_nasality: float, mean_height: float, s_nasality: float, s_height: float,
                     c_nasality_height: float):
        """
        Defines self.mean (1d list): self.mean[0] = mean_nasality, self.mean[1] = mean_height
        Defines self.cov (2d list): covariance matrix for height and nasality
        :param mean_nasality: float,mean of this category on nasality dimension
        :param mean_height: float, mean of this category on height dimension
        :param s_nasality: float,variance on nasality dimension
        :param s_height: float,variance on height dimension
        :param c_nasality_height: float, covariance of nasality and height
        """
        mean = [mean_nasality, mean_height]
        cov = [[s_nasality, c_nasality_height], [c_nasality_height, s_height]]
        return cls(mean, cov)

    def sample(self, num_samples, seed):
        """Returns num_samples (int) number of random samples from the distribution defined by
         self.mean and self.cov (in numpy array)
        num_samples: number of samples per category
        seed: random seed for sample generation
        Returned random samples are a num_samples x 2 numpy array
        """
        # print(self.cov)
        np.random.seed(seed)
        return np.random.default_rng(seed=seed).multivariate_normal(self.mean, self.cov, num_samples)

    def shift_sample(self, num_samples:int, seed:int, bias:float):
        """
        Returns num_samples number of random samples from the distribution defined by self.mean and
        self.cov with each sample along the nasality dimension shifted by "bias" amount
        :param num_samples: number of samples
        :param seed: random seed for sample generation
        :param bias: size of bias
        :return: numpy array of shifted samples
        """
        sample = self.sample(num_samples, seed)
        sample[:, 0] += bias
        return sample

    def threshold_shift_sample(self, num_samples:int, seed:int, shift:float, threshold:float):
        """

        Returns num_samples number of random samples from the distribution defined by self.mean and
        self.cov with each sample along the nasality dimension shifted by "bias" amount, if the sample's
        height is below threshold
        :param num_samples: number of samples
        :param seed: random seed for sample generation
        :param shift: size of bias
        :param threshold height value resulting in a biased nasality sample
        :return: numpy array of shifted samples
        """
        sample = self.sample(num_samples, seed)
        boolean_mask = sample[:,1] < threshold #Get heights lower than threshold
        sample[boolean_mask, 0] += shift # Add bias to nasality for the samples with those heights
        return sample
        # if self.mean[1] < threshold:
        #     return self.shift_sample(num_samples, seed=seed, bias=bias)
        # else:
        #     return self.sample(num_samples, seed=seed)


    def __str__(self):
        return "Mean:" + str(self.mean) + " " + "Covariance:" + str(self.cov)


class Language:
    """
    Each LanguageInput object represents a
    system of Gaussian vowel categories defined on height and nasality dimensions.
    A LanguageInput consists of a list of Category objects, self.vowels, and methods
    on that list of Category objects:
    sampling from them (sample),
    """

    def __init__(self, vowels:list=None, name:str=None, priors:list = None):
        """
        Initializes LanguageInput with vowel categories
        :param vowels If vowels is None, initializes LanguageInput with empty list of vowel categories
        :param name optional string name for labeling this language
        :param priors list of relative frequency of each vowel category, in order based on the order of self.vowels.
         If None, set to 1 for each category
        """
        self.vowels = vowels if vowels else []
        self.priors = priors
        if self.priors is None:
            self.priors = [1] * len(vowels)
        self.name = name

    def sample(self, num_samples_per_cat, seed, get_labels=False, shift=None, threshold=None, shift_categories=None, scale=False):
        """
        Samples num_samples_per_cat vowels per vowel category in self.vowels
        :param num_samples_per_cat: int, number of samples to generate of each vowel category
        :param seed: int, random seed to use in sampling
        :param get_labels: Boolean, whether to return a data structure with category labels
        :param shift: If not None, the amount to shift nasality samples by, if they meet threshold
        :param threshold: If not None, the height value required of a sample for its nasality values to be biased
        :param shift_categories: If not None, the list of category names (str) that should be shifted by shift amount
        :param scale: If True, scale the number of samples per category by its corresponding value in self.priors
        :return: numpy array of shape num_samples_per_cat * number of categories x 2;
         if labels, also return array of shape num samples per cat * number of categories * 1 - a list where each item
         is the label of the category the corresponding sample was generated from

        """
        assert (shift and threshold) or (not shift and not threshold), "Shift and threshold must both or neither be None"

        if scale: #TODO: integrate priors/scale with learner
            adjusted_num_samples_per_cat = [math.floor(num_samples_per_cat * len(self.vowels) * prior) for prior in self.priors]
        else:
            adjusted_num_samples_per_cat = [num_samples_per_cat for vowel in self.vowels]

        if threshold:
            samples = np.concatenate(tuple(
                [vowel.threshold_shift_sample(adjusted_num_samples_per_cat[index], seed=seed, threshold=threshold, shift=shift) for
                 index,vowel in enumerate(self.vowels)]))
        elif shift_categories:
            #Todo: test category shift
            cat_samples = []
            for index,vowel in enumerate(self.vowels):
                if vowel.name in shift_categories:
                    cat_samples.append(vowel.threshold_shift_sample(adjusted_num_samples_per_cat[index], seed=seed, threshold=threshold, shift=shift))
                else:
                    cat_samples.append(vowel.sample(adjusted_num_samples_per_cat[index], seed=seed))
            samples = np.concatenate(tuple(cat_samples))
        else:
            #samples = np.concatenate(tuple([vowel.sample(num_samples_per_cat, seed=seed) for vowel in self.vowels]))
            samples = np.concatenate(tuple([vowel.sample(adjusted_num_samples_per_cat[index], seed=seed) for index,vowel in enumerate(self.vowels)]))
        if not get_labels:
            return samples
        #label_array = np.concatenate(tuple([[index] * num_samples_per_cat for index, vowel in enumerate(self.vowels)]))
        label_array = np.concatenate(tuple([[index] * adjusted_num_samples_per_cat[index] for index, vowel in enumerate(self.vowels)]))
        if debug:
            print("Labels:", get_labels)
        return samples, label_array

    def plot_categories(self, savefilename=None, title=None, xrange=(-10, 10), yrange=(-10, 10), showSamples=False, seed=1):
        """
        Plots an ellipse based on each category's mean and covariance
        Each category is shown in a different color (up to 5 colors)
        Based on the Gaussian Mixture plotting example on scikit learn's website
        :param savefilename If not None, the string filename to save the plot to instead of displaying during execution
        :param title If not None, the string title to put on the plot. If None, uses self.name. If self.name is None, plot is untitled
        :param xrange: float tuple, x-axis range of plot (nasality): (min, max)
        :param yrange: float tuple, y-axis range of plot (height): (min, max)
        :param showSamples: boolean, whether to show samples from categories
        :param seed: seed for sampling datapoints if showSamples
        :return: None
        """


        # Colors to go through
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

        # Means and covariances of each category
        means, covariances = list(zip(*[(cat.mean, cat.cov) for cat in self.vowels]))

        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            if showSamples:
                X = np.random.default_rng(seed).multivariate_normal(mean, covar, size=int(100*self.priors[i]))
                #print(mean, X)
                plt.scatter(X[:,0],X[:,1], color=color, alpha=self.priors[i])
            # ell.set_clip_box(splot.bbox)
            ax = plt.gca()
            ell.set_alpha(self.priors[i])
            ax.add_patch(ell)

        plt.xlim(xrange[0], xrange[1])
        plt.ylim(yrange[0], yrange[1])
        plt.xticks()
        plt.yticks()
        if title:
            plt.title(title)
        elif self.name:
            plt.title(self.name)
        if savefilename:
            plt.savefig(savefilename)
            plt.close()
        else:
            plt.show()

    def __str__(self):
        return "\n".join([str(vowel) + "\n" for vowel in self.vowels])


class Learner:
    """
    A Learner has:
        An input Language object (self.inputLanguage)
        A scale setting: whether sampling from the input language should be scaled by its prior for each of its categories
        A learned Language object (self.learnedLanguage)
        A fitted SciKitLearn Bayesian mixture learner (self.dpgmm)
        An ARI score against the input language (self.ari), after evaluate_accuracy has been called
        An ARI score based only on input language category mean height values (self.height_ari), after evaluate_height_categories_ari has been called
        An effective number of categories - how many are used to predict the training data, maximum a posteriori (self.effective_cats), after effective_categories has been called
        A dictionary count of how many training samples are predicted to belong to each category (self.category_counts), after effective_categories has been called
        Optional: a bias to shift samples by (self.contextless_bias) if they fall below a given height value (self.threshold)
        A set of hyperparameter settings:
            -number of samples per category from input language (integer greater than 0): self.numSamples
            -weight concentration parameter for Dirichlet prior(float): self.concentration
            -maximum number of learned categories (integer greater than 0): self.maxCats
            -covariance type (default full) self.covType
            -parameter initialization (default kmeans): self.paramInit
            -random seed (default 1): self.seed
    Wrapper for SciKitLearn's Bayesian mixture learner, for running vowel inventory simulations
    """

    def __init__(self, inputLanguage: Language,
                 numSamples: int = 100,
                 scale: bool = False,
                 concentration: float = None,
                 maxCats: int = 6,
                 covType: str = 'full',
                 param_init: str = 'kmeans',
                 max_iters = 100,
                 seed: int = 1,
                 name: str = None,
                 contextless_bias: float = None,
                 threshold: float = None):
        """
        Initializes a Learner object with the following properties
        :param inputLanguage: Language object that generates the learning input
        :param numSamples: number of samples in learning input from each of the inputLanguage's categories
        :param scale: whether to scale numSamples for each category by its prior in inputLanguage
        :param concentration: concentration parameter for Dirichlet prior (If None, then 1/maxCats - see SciKitLearn BayesianMixture)
        :param maxCats: maximum number of learned categories (greater than 1)
        :param covType: covariance type (see SciKit Learn's covariance type options for Bayesian mixture)
        :param param_init: learner parameter initialization (see SciKit Learn's param initialization options for Bayesian mixture)
        :param max_iters: maximum number of iterations allowed to reach convergence
        :param seed:  random seed
        :param name: optional string name for this learner
        :param contextless_bias: Value to shift learning input samples by along the nasality dimension, if they fall below the height threshold
        :param threshold: Value where learning input samples with a height below this value are shifted in nasality by bias

        Runs the learner with the input language and creates the following properties:
        dpgmm: BayesianGaussianMixture object fit to samples from the inputLanugae (numSamples samples per category)
        self.learnedLanguage: Language object consisting of the vowel categories learned

        """
        self.inputLanguage = inputLanguage
        self.numSamples = numSamples
        self.scale = scale
        self.concentration = concentration
        self.maxCats = maxCats
        self.covType = covType
        self.param_init = param_init
        self.max_iters = max_iters
        self.seed = seed
        self.name = name
        self.contextless_bias = contextless_bias
        self.threshold = threshold
        self.effective_cats = None
        self.category_counts = None
        self.ari = None
        self.height_ari = None


        # Generate samples from input language

        #Perception bias
        if self.contextless_bias:
            self.samples, self.labels = self.inputLanguage.sample(self.numSamples, seed=self.seed, shift=self.contextless_bias, threshold=self.threshold, get_labels=True, scale=self.scale)
        #No perception bias
        else:
            self.samples, self.labels = self.inputLanguage.sample(self.numSamples, seed = self.seed, get_labels=True, scale=self.scale)

        # Fit model
        self.dpgmm = mixture.BayesianGaussianMixture(weight_concentration_prior=self.concentration,
                                                     n_components=self.maxCats,
                                                     covariance_type=self.covType,
                                                     init_params=self.param_init,
                                                     max_iter=self.max_iters,
                                                     random_state=self.seed
                                                     )
        self.dpgmm.fit(self.samples)


        # Format learned categories
        learned_vowels = [Category(mean, self.dpgmm.covariances_[index]) for index, mean in
                          enumerate(self.dpgmm.means_)]
        self.learnedLanguage = Language(learned_vowels, priors=self.dpgmm.weights_)

        if debug:
            print("Model name:", name)
            print("Model means ", self.dpgmm.means_)

    def evaluate_accuracy(self, samples=None, labels=None):
        """
        If self.ari is not None, re-uses that value. Otherwise:
        Evaluates the Adjusted Rand Index for this model's predictions versus a set of 'correct' cluster labels.
        If samples and labels are both none, tests on a sample from self.inputLanguage, taking self.numSamples number
        of samples from each category in self.inputLanguage.
        :param samples: None, or 2-D array of height, nasality values (one entry per sample)
        :param labels: None, or 1-D array of cluster labels: correct sample labels to compare against. Must be same length as samples
        :return: the ARI for this learner's predictions
        """
        if self.ari: #Just re-use this value if already computed
            return self.ari
        if not samples:
            #samples, labels = self.inputLanguage.sample(self.numSamples, seed=self.seed, get_labels=True)
            samples = self.samples
            labels = self.labels #Re-use samples to save computation
        predicted_labels = self.dpgmm.predict(samples)
        ari = adjusted_rand_score(labels, predicted_labels)
        if debug:
            print("Predicted", predicted_labels)
            print("Actual", labels)
            print("ARI", ari)
        self.ari = ari
        return self.ari

    def evaluate_height_categories_ari(self):
        """
        Returns the ARI between model predicted categories and the categories predicted by the input language, modified so
        to ignore nasality (all input language categories set to have the mean same nasality)
        Assumes that ideal language has uniform prior over categories
        :return: float
        """
        #TODO: test this function
        #Evaluate against input language's height categories, equalizing nasal means to average
        average_nasality = np.mean([vowel.mean[0] for vowel in self.inputLanguage.vowels])
        flat_nasal_input_vowels = [Category(vowel.mean, vowel.cov) for vowel in self.inputLanguage.vowels]
        for vowel in flat_nasal_input_vowels:
            vowel.mean[0] = average_nasality
        input_height_cats = [multivariate_normal(mean=cat.mean, cov=cat.cov) for cat in flat_nasal_input_vowels]

        # Get index of category that has the highest likelihood for each sample
        max_likelihood_labels = [
            max(*[(index, cat.pdf(sample)) for index, cat in enumerate(input_height_cats)], key=lambda item: item[1]) for sample in
            self.samples]
        input_height_labels = [max_likelihood_label[0] for max_likelihood_label in max_likelihood_labels]

        #Get predicted categories from learned model
        model_labels = self.dpgmm.predict(self.samples)

        return adjusted_rand_score(labels_true=input_height_labels, labels_pred=model_labels)

    def effective_categories(self):
        """
        Returns only the categories in self.learnedLanguage predicted as labels for a sample from self.inputLanguage
        Also computes the number of samples (given this.inputLanguage and this.seed) predicted to be in each category (by maximum probability)
        :return: list of Category
        """
        if self.effective_cats: #Just reuse this value if already computed
            return self.effective_cats

        #sample = self.inputLanguage.sample(self.numSamples, seed=self.seed, shift=self.contextless_bias, threshold=self.threshold)
        sample = self.samples
        predicted_label = self.dpgmm.predict(sample)
        means = self.dpgmm.means_
        covs = self.dpgmm.covariances_
        category_counts = {}
        categories = []
        for index, (mean, cov) in enumerate(zip(means, covs)):
            category_counts[index] = np.sum(predicted_label == index)
            if not np.any(predicted_label == index):
                continue
            categories.append(Category(mean, cov))
        self.category_counts = category_counts
        self.effective_cats = categories
        return self.effective_cats



    def plot_predictions(self, xrange=(-10, 10), yrange=(-10, 10),title=None, savefilename=None):
        """
        Creates a color-labeled scatterplot of self.dpgmm's predicted categorizations
        on a sample from self.inputLanguagePlots using self.seed and self.numSamples
        as well as an ellipse based on each category's mean and covariance
        Each category is shown in a different color (up to 5 colors)
        Based on the Gaussian Mixture plotting example on scikit learn's website
        :param xrange tuple of minimum x axis value, maximum x axis value
        :param yrange tuple of minimum y axis value, maximum y axis value
        :param title If not None, the string title to put on the plot. If None, uses self.name. If self.name is None, plot is untitled
        :param savefilename If not None, the string filename to save the plot to, instead of displaying during execution
        :return: None
        """
       # X = self.inputLanguage.sample(self.numSamples, seed = self.seed)
        X = self.samples
        Y_ = self.dpgmm.predict(X)
        means = [vowel.mean for vowel in self.learnedLanguage.vowels]
        #print(Y_)
        covariances = [vowel.cov for vowel in self.learnedLanguage.vowels]
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
        for i, mean in enumerate(means):
            v, w = linalg.eigh(covariances[i])
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=next(color_iter))

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=next(color_iter))
            # ell.set_clip_box(splot.bbox)
            ax = plt.gca()
            ell.set_alpha(self.dpgmm.weights_[i])
            ax.add_patch(ell)

        plt.xlim(xrange[0], xrange[1])
        plt.ylim(yrange[0], yrange[1])
        plt.xticks()
        plt.yticks()
        if title:
            plt.title(title)
        elif self.name:
            plt.title(self.name)

        if savefilename:
            plt.savefig(savefilename)
            plt.close()
        else:
            plt.show()

    def __str__(self):
        converged = "converged" if self.dpgmm.converged_ else "not converged"
        num_cats = "Number of categories:" + str(len(self.effective_categories()))
        categories = "\n".join([str(cat.mean) for cat in self.effective_categories()])
        properties = [self.name, converged, num_cats, categories] if self.name else [converged, num_cats, categories]
        return "\n".join(properties)



# Informal tests as we go
if __name__ == "__main__":
    cat_1 = Category.build_params(mean_nasality=1.5, mean_height=5, s_nasality=1, s_height=1, c_nasality_height=.5)
    cat_2 = Category.build_params(mean_nasality=3, mean_height=-5, s_nasality=1, s_height=1, c_nasality_height=.5)
    simple_lang = Language(vowels=[cat_1, cat_2])
    simple_lang.plot_categories()
    simple_learner = Learner(inputLanguage=simple_lang, name="first")
    simple_learner.learnedLanguage.plot_categories()
    print(simple_learner.dpgmm.weights_)
    # print(simple_learner.learnedLanguage)
    # print(len(simple_learner.dpgmm.means_))
    # simple_learner_2 = Learner(inputLanguage=simple_learner.learnedLanguage, name="second")
    # print(simple_learner_2.learnedLanguage)
    # print(len(simple_learner_2.dpgmm.means_))
