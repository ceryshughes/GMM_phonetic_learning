# Program defining the Language, Learner, Category classes and the ARI and plot methods
# General structures used for running vowel inventory evolution simulations
import numpy as np
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from scipy import linalg

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
        Defines this.mean (1d list): this.mean[0] = mean_nasality, this.mean[1] = mean_height
        Defines this.cov (2d list): covariance matrix for height and nasality
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
         self.mean and self.cov
        num_samples: number of samples per category
        seed: random seed for sample generation
        Returned random samples are a num_samples x 2 numpy array
        """
        # print(self.cov)
        np.random.seed(seed)
        return np.random.default_rng().multivariate_normal(self.mean, self.cov, num_samples)

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

    def __init__(self, vowels=None, name=None):
        """
        Initializes LanguageInput with vowel categories
        :param vowels If vowels is None, initializes LanguageInput with empty list of vowel categories
        :param name optional string name for labeling this language
        """
        self.vowels = vowels if vowels else []
        self.name = name

    def sample(self, num_samples_per_cat, seed, get_labels=False):
        """
        Samples num_samples_per_cat vowels per vowel category in self.vowels
        :param num_samples_per_cat: int, number of samples to generate of each vowel category
        :param seed: int, random seed to use in sampling
        :param get_labels: Boolean, whether to return a data structure with category labels
        :return: numpy array of shape num_samples_per_cat * number of categories x 2;
         if labels, also return array of shape num samples per cat * number of categories * 1

        """
        samples = np.concatenate(tuple([vowel.sample(num_samples_per_cat, seed=seed) for vowel in self.vowels]))
        if debug:
            print("Samples:", samples)
        if not get_labels:
            return samples
        label_array = np.concatenate(tuple([[index] * num_samples_per_cat for index, vowel in enumerate(self.vowels)]))
        if debug:
            print("Labels:", get_labels)
        return samples, label_array

    def plot_categories(self, savefilename=None, title=None, xrange=(-10, 10), yrange=(-10, 10)):
        """
        Plots an ellipse based on each category's mean and covariance
        Each category is shown in a different color (up to 5 colors)
        Based on the Gaussian Mixture plotting example on scikit learn's website
        :param savefilename If not None, the string filename to save the plot to
        :param title If not None, the string title to put on the plot. If None, uses self.name. If self.name is None, plot is untitled
        :param xrange: float tuple, x-axis range of plot (nasality): (min, max)
        :param yrange: float tuple, y-axis range of plot (height): (min, max)
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
            if debug:
                X = np.random.default_rng().multivariate_normal(mean, covar, 100)
                print(mean, X)
                plt.scatter(X[:,0],X[:,1], color=color)
            # ell.set_clip_box(splot.bbox)
            ax = plt.gca()
            ell.set_alpha(0.5)
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
        plt.show()

    def __str__(self):
        return "\n".join([str(vowel) + "\n" for vowel in self.vowels])


class Learner:
    """
    A Learner has:
        An input Language object
        XXXXXA fit method
        A learned Language object (after the fit method has been run)
        A predict method, based ont he learned Language object (after the fit method has been run)
        A set of hyperparameter settings:
            -number of samples per category from input language (integer greater than 0)
            -weight concentration parameter for Dirichlet prior(float)
            -maximum number of learned categories (integer greater than 0)
            -covariance type (default full)
            -parameter initialization (default kmeans)
            -random seed (default 1)
    Wrapper for SciKitLearn's Bayesian mixture learner, for running vowel inventory simulations
    """

    def __init__(self, inputLanguage: Language,
                 numSamples: int = 100,
                 concentration: float = None,
                 maxCats: int = 6,
                 covType: str = 'full',
                 param_init: str = 'kmeans',
                 max_iters = 100,
                 seed: int = 1,
                 name: str = None):
        """
        Initializes a Learner object with the following properties
        :param inputLanguage: Language object that generates the learning input
        :param numSamples: number of samples in learning input from each of the inputLanguage's categories
        :param concentration: concentration parameter for Dirichlet prior (If None, then 1/maxCats - see SciKitLearn BayesianMixture)
        :param maxCats: maximum number of learned categories (greater than 1)
        :param covType: covariance type (see SciKit Learn's covariance type options for Bayesian mixture)
        :param param_init: learner parameter initialization (see SciKit Learn's param initialization options for Bayesian mixture)
        :param max_iters: maximum number of iterations allowed to reach convergence
        :param seed:  random seed
        :param name: optional string name for this learner

        Runs the learner with the input language and creates the following properties:
        dpgmm: BayesianGaussianMixture object fit to samples from the inputLanugae (numSamples samples per category)
        self.learnedLanguage: Language object consisting of the vowel categories learned

        """
        self.inputLanguage = inputLanguage
        self.numSamples = numSamples
        self.concentration = concentration
        self.maxCats = maxCats
        self.covType = covType
        self.param_init = param_init
        self.max_iters = max_iters
        self.seed = seed
        self.name = name


        # Generate samples from input language
        samples = self.inputLanguage.sample(self.numSamples, seed = self.seed)

        # Fit model
        self.dpgmm = mixture.BayesianGaussianMixture(weight_concentration_prior=self.concentration,
                                                     n_components=self.maxCats,
                                                     covariance_type=self.covType,
                                                     init_params=self.param_init,
                                                     max_iter=self.max_iters,
                                                     random_state=self.seed
                                                     )
        self.dpgmm.fit(samples)

        # Format learned categories
        learned_vowels = [Category(mean, self.dpgmm.covariances_[index]) for index, mean in
                          enumerate(self.dpgmm.means_)]
        self.learnedLanguage = Language(learned_vowels)

        if debug:
            print("Model name:", name)
            print("Model means ", self.dpgmm.means_)

    def evaluate_accuracy(self, samples=None, labels=None):
        """
        Evaluates the Adjusted Rand Index for this model's predictions versus a set of 'correct' cluster labels.
        If samples and labels are both none, tests on a sample from self.inputLanguage, taking self.numSamples number
        of samples from each category in self.inputLanguage
        :param samples: None, or 2-D array of height, nasality values (one entry per sample)
        :param labels: None, or 1-D array of cluster labels: correct sample labels to compare against. Must be same length as samples
        :return: the ARI for this learner's predictions
        """
        if not samples:
            samples, labels = self.inputLanguage.sample(self.numSamples, seed=self.seed, get_labels=True)
        predicted_labels = self.dpgmm.predict(samples)
        ari = adjusted_rand_score(labels, predicted_labels)
        if debug:
            print("Predicted", predicted_labels)
            print("Actual", labels)
            print("ARI", ari)
        return ari

    def effective_categories(self):
        """
        Returns only the categories in self.learnedLanguage predicted as labels for a sample from self.inputLanguage
        :return: list of Category
        """
        sample = self.inputLanguage.sample(self.numSamples, seed=self.seed)
        predicted_label = self.dpgmm.predict(sample)
        means = self.dpgmm.means_
        covs = self.dpgmm.covariances_
        categories = []
        for index, (mean, cov) in enumerate(zip(means, covs)):
            if not np.any(predicted_label == index):
                continue
            categories.append(Category(mean, cov))
        return categories



    def plot_predictions(self, xrange=(-10, 10), yrange=(-10, 10),title=None):
        """
        Creates a color-labeled scatterplot of self.dpgmm's predicted categorizations
        on a sample from self.inputLanguagePlots
        as well as an ellipse based on each category's mean and covariance
        Each category is shown in a different color (up to 5 colors)
        Based on the Gaussian Mixture plotting example on scikit learn's website
        :param xrange tuple of minimum x axis value, maximum x axis value
        :param yrange tuple of minimum y axis value, maximum y axis value
        :param title If not None, the string title to put on the plot. If None, uses self.name. If self.name is None, plot is untitled
        :return: None
        """
        X = self.inputLanguage.sample(self.numSamples, seed = self.seed)
        Y_ = self.dpgmm.predict(X)
        means = [vowel.mean for vowel in self.learnedLanguage.vowels]
        covariances = [vowel.cov for vowel in self.learnedLanguage.vowels]
        color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
            # ell.set_clip_box(splot.bbox)
            ax = plt.gca()
            ell.set_alpha(0.5)
            ax.add_patch(ell)

        plt.xlim(xrange[0], xrange[1])
        plt.ylim(yrange[0], yrange[1])
        plt.xticks()
        plt.yticks()
        if title:
            plt.title(title)
        elif self.name:
            plt.title(self.name)
        plt.show()




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
