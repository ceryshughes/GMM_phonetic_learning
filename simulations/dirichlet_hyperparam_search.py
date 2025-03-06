# Python file for a hyperparameter search on SciKit Learn's Bayesian Mixture of Gaussians model
import itertools
import math
import os

import scipy
import scipy.linalg

from learner_definitions import Category, Language, Learner
import numpy as np
import csv
from collections import defaultdict
import pathlib
from sklearn import mixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal

debug = False

def run_across_seeds(
                     inputLanguage:Language,
                     numSamples:int = 100,
                     concentration:float = None,
                     maxCats:int = 6,
                     max_iters:int = 100,
                     numSeeds = 10):
    """
    Runs Learners with identical parameters, just with different random seeds
    :param inputLanguage: Input Language for the learner
    :param numSamples:  number of samples per category given to the learner
    :param concentration: Dirichlet prior concentration parameter
    :param maxCats: maximum number of categories available to the learner
    :param max_inters: maximum number of iterations learner can run
    :param numSeeds: number of random seeds to try
    :return: A list of Learner objects, one for each seed in seeds (in order)
    """
    seeds = [i for i in range(0, numSeeds)]
    if debug:
        print(concentration, numSamples, maxCats, max_iters)
        seeds = [i for i in range(0,10)]
    learners = []
    for seed in seeds:
        learner = Learner(seed=seed,
                          inputLanguage=inputLanguage,
                          numSamples=numSamples,
                          concentration=concentration,
                          maxCats=maxCats,
                          max_iters=max_iters)
        learners.append(learner)
        if debug:
            print(str(learner))
            print()
    return learners


# Define easy language: 4 nonoverlapping categories
def define_easy_language():
    """
    Sets up a simple baseline language that should be trivial for a cluster learner
    :return: a Language with 4 separable categories, named "trivial4" with the language.name property
    """
    cat_1 = Category.build_params(mean_nasality=.5, mean_height=.5, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_2 = Category.build_params(mean_nasality=.5, mean_height=8, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_3 = Category.build_params(mean_nasality=8, mean_height=.5, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_4 = Category.build_params(mean_nasality=8, mean_height=8, s_nasality=.5, s_height=.5, c_nasality_height=0)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="trivial4")

def define_slight_overlap_language():
    """
        Sets up a simple baseline language that should be easy for a cluster learner
        :return: a Language with 4 separable categories, named "slightoverlap4" with the language.name property
        """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_2 = Category.build_params(mean_nasality=1, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_3 = Category.build_params(mean_nasality=4, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_4 = Category.build_params(mean_nasality=4, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="slightoverlap4")

def define_debug_language():
    """
          Sets up a two-category language for debugging
          :return: a Language with 2 separable categories, named "debuglang" with the language.name property
          """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_2 = Category.build_params(mean_nasality=4, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    return Language(vowels=[cat_1, cat_2], name="debuglang")

def define_scikit_example_language():
    #Sanity check against the demonstration in https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html#sphx-glr-auto-examples-mixture-plot-concentration-prior-py
    cat_1 = Category.build_params(mean_nasality=0, mean_height=-0.70, s_nasality=.7, s_height=.1, c_nasality_height=0)
    cat_2 = Category.build_params(mean_nasality=0, mean_height=0, s_nasality=.5, s_height=.1, c_nasality_height=0)
    cat_3 = Category.build_params(mean_nasality=0, mean_height=0.70, s_nasality=.5, s_height=.1, c_nasality_height=0)
    return Language(vowels=[cat_1, cat_2], name="scikitlang")

def define_challenge_language():
    """
          Sets up a language that should be harder for a cluster learner because the distributions are overlapping
          :return: a Language with 4 categories, named "challenge" with the language.name property
          """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_2 = Category.build_params(mean_nasality=1, mean_height=4, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_3 = Category.build_params(mean_nasality=4, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_4 = Category.build_params(mean_nasality=4, mean_height=4, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="challenge")

def define_challenge_language_2():
    """
          Sets up a language that should be even harder for a cluster learner because the distributions are more overlapping
          than the language generated by define_challenge_language()
          :return: a Language with 4 categories, named "challenge2" with the language.name property
          """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_2 = Category.build_params(mean_nasality=1, mean_height=4, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_3 = Category.build_params(mean_nasality=2.5, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    cat_4 = Category.build_params(mean_nasality=2.5, mean_height=4, s_nasality=1.5, s_height=1.5, c_nasality_height=.5)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="challenge2")

def define_challenge_sphere():
    """
            Like the language generated by define_challenge_language_2, but without covariation between height and nasality
            :return: a Language with 4 categories, named "challenge_sphere" with the language.name property
            """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=0)
    cat_2 = Category.build_params(mean_nasality=1, mean_height=3, s_nasality=1.5, s_height=1.5, c_nasality_height=0)
    cat_3 = Category.build_params(mean_nasality=3, mean_height=1, s_nasality=1.5, s_height=1.5, c_nasality_height=0)
    cat_4 = Category.build_params(mean_nasality=3, mean_height=3, s_nasality=1.5, s_height=1.5, c_nasality_height=0)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4], name="challenge_sphere")

def nasal_split(cat1: Category, cat2: Category):
    # Check whether category 1 and 2 have similar heights, but different nasality
    same_height = abs(cat1.mean[1] - cat2.mean[1]) < 1
    diff_nasality = abs(cat1.mean[0] - cat2.mean[0]) > .75 #tighter range for nasality
    return same_height and diff_nasality

def nasal_splits(cats: list):
    #For a list of categories, check if each pair has similar height but different nasality
    #Returns the number of pairs that are split for nasality (similar height but different nasality)
    #the number of those split pairs that have a high height, the number that has a low height,
    #and the number whose height is neither high or low
    num_nasal_splits = 0
    high_splits = 0
    low_splits = 0
    other_splits = 0
    for cat_pair in itertools.combinations(cats,2):
        #print(cat_pair)
        if nasal_split(cat_pair[0], cat_pair[1]):
            num_nasal_splits += 1
            if cat_pair[0].mean[1] > 3: # High vowel
                high_splits += 1 #TODO: get rid of magic numbers
            elif cat_pair[0].mean[1] < 2: #Low vowel
                low_splits +=1
            else:
                other_splits += 1
    return num_nasal_splits, high_splits, low_splits, other_splits






def output_to_csv(runs: dict, filename: str, run_col:str):
    """

    :param runs: Dictionary of [string -> dictionary of [string -> string, float, or int]]where:
                -each key is a string description of the hyperparameter setting being tried
                -each value is a dictionary of string -> string, float or int where:
                    -each key is the name of a metric or other recorded value
                    -each value is the value for that metric for this hyperparameter setting
                    -all the values share the same set of keys
    :param filename: the filename to save the csv to
    :param run_col: column header for hyperparameter setting
    :return:
    """
    file = open(filename, "w+", newline="")
    csv_writer = csv.DictWriter(file, fieldnames=[run_col]+list(list(runs.values())[0].keys()))
    csv_writer.writeheader()
    for setting in runs:
        row = runs[setting]
        row.update({run_col:setting})
        csv_writer.writerow(row)

    file.close()


def output_learners(filename:str, learners:list, seeds_to_plot: list = None, plot_dir = ""):
    """

    :param filename: name of file to append individual learner information to
    :param learners: list of Learner objects
    :param seeds_to_plot: list of seeds (int); plot the predictions of learners that have these seed
    :param plot_dir: string name of directory to save plots in

    :return: None
    """
    seeds_to_plot = seeds_to_plot if seeds_to_plot else []
    need_header = not pathlib.Path(filename).exists()
    file = open(filename, "a", newline="")
    writer = csv.DictWriter(file, fieldnames=["Concentration",
                                              "Seed",
                                              "NumSamples",
                                              "MaxIters",
                                              "MaxCats",
                                              "Converged",
                                              "ARI",
                                              "HeightARI",
                                              "NumCats", "NumBigCats",
                                              "Means", "Weights",
                                              "Covs", "NumLargeWeights", "CategoryCounts",
                                              "NumNasalSplits",
                                              "NumHighSplits",
                                              "NumLowSplits"
                                              ])
    if need_header:
        writer.writeheader()
    for learner in learners:
        effective_cats = learner.effective_categories()
        num_splits, high_splits, low_splits, other_splits = nasal_splits(effective_cats)
        row = {"Concentration":learner.concentration,
               "Seed":learner.seed,
               "NumSamples": learner.numSamples,
                "MaxIters": learner.max_iters,
                "MaxCats": learner.maxCats,
                "Converged":learner.dpgmm.converged_,
                "ARI": learner.evaluate_accuracy(),
               "HeightARI": learner.evaluate_height_categories_ari(),
                "NumCats":len(effective_cats),
                "NumBigCats": len([count for count in learner.category_counts.values() if count > 5]),
                "Means": " ".join([str(cat.mean) for cat in effective_cats]).replace("\n"," "),
                "Weights": str(learner.dpgmm.weights_).replace("\n", ""),
               "Covs": " ".join([str(cat.cov) for cat in effective_cats]).replace("\n", " "),
               "NumLargeWeights":str(len([weight for weight in learner.dpgmm.weights_ if weight > 0.001])),
               "CategoryCounts": str(learner.category_counts),
               "NumNasalSplits":str(num_splits),
               "NumHighSplits":str(high_splits),
               "NumLowSplits":str(low_splits)}
        if learner.seed in seeds_to_plot:
            filename = plot_dir + "/" + "_".join([learner.inputLanguage.name if learner.inputLanguage.name else "",
                                                  str(learner.seed), str(learner.concentration),
                                                 str(learner.numSamples), str(learner.max_iters),
                                                  str(learner.maxCats), str(learner.dpgmm.converged_)]) + ".pdf"
            title = " ".join([str(learner.seed),
                             learner.inputLanguage.name if learner.inputLanguage.name else "",
                             str(learner.concentration),
                              str(learner.max_iters),
                              str(learner.maxCats),
                              str(len(effective_cats))])
            learner.plot_predictions(savefilename=filename, title=title)

       # if learner.seed == 17:
       #     learner.plot_predictions()
        writer.writerow(row)
    file.close()



# def concentration_trial_param_search(inputLanguage:Language, outputFile:str, concentration_trials:list, maxCats:int, numSamples:int):
#     if debug:
#         inputLanguage.plot_categories()
#
#     seeds = [i for i in range(0,10)]
#     concentration_run_data = defaultdict(lambda: {})
#     for trial in concentration_trials:
#         learners = run_across_seeds(seeds,
#                                     inputLanguage=inputLanguage,
#                                     concentration=trial, maxCats=maxCats, numSamples=numSamples)
#         # concentration_results[trial]
#         # concentration_aris[trial] = np.mean([learner.evaluate_accuracy() for learner in learners])
#         aris = [learner.evaluate_accuracy() for learner in learners]
#
#         converged = [1 if learner.dpgmm.converged_ else 0 for learner in learners]
#         if debug:
#             learners[0].plot_predictions()
#             print(sorted([weight for weight in learners[0].dpgmm.weights_]))
#             num_cats = [len(learner.learnedLanguage.vowels) for learner in
#                         learners]  # TODO: This will always be the max number! Need to look at priors on each cat
#             #exit()
#
#         effective_cat_counts = [len(learner.effective_categories()) for learner in learners]
#         concentration_run_data[trial] = {
#             "Mean ARI": np.mean(aris),
#             "ARI standard dev": np.std(aris),
#             "Percent converged": sum(converged) / len(converged),
#             "Mean number of categories": np.mean(effective_cat_counts),
#             "Number of categories standard dev": np.std(effective_cat_counts)
#         }
#     output_to_csv(concentration_run_data, outputFile, "Concentration")

#note: Bayesian modeling perspective: some just optimization stuff, some interesting modeling predictions (prior, amount of data)
def param_search(inputLanguage:Language,
                 outputFile:str,
                 concentration_trials:list = None,
                 concentration:float = None,
                 maxCatsList: list = None,
                 maxCats:int = None,
                 numSamples:int = None,
                 numSamplesList: list = None,
                 maxItersList: list = None,
                 maxIters:int = 100,
                 numSeeds=10,
                 view_learners = False,
                 seeds_to_view = []):
    """
        -Runs a Learner across the desired parameter settings (varying either concentration prior, maximum number of categories, or
        number of samples per category taken from input language)
        -Tries 100 random seeds for each setting
        -Outputs the results (ARIs, convergence, number of effective categories) to a csv file, as well as information about the run (hyperparameter settings and number of seeds)
        :param inputLanguage: language to train the learners on
        :param outputFile: name of the csv output file to put the results in
        :param concentration_trials: If trying different concentration values, the concentration values to try. Otherwise None
        :param concentration: If not trying different concentration values, the concentration value to use
        :param maxCatsList: If trying different maximum category numbers, the maximum category numbers to try. Otherwise None
        :param maxCats If not trying different maximum category values, the maximum category value to use
        :param numSamples: If not trying different numbers of samples, the number of samples to use.
        :param numSamplesList: If trying different numbers of samples, the numbers of samples to try. Otherwise None
        :param maxItersList: If trying different numbers of maximum iterations, the numbers to try. Otherwise None
        :param maxIters: maximum number of learning iterations. Default 100 if not specified and maxItersList not specified
        :param numSeeds: number of seeds to try per hyperparameter value
        :param view_learners: whether to output additional csv file with information about each individual learner
        :param seeds_to_view: If view_learners is defined, list of seeds whose learners' predictions should be plotted
    """
    assert (concentration_trials and maxCats and numSamples and maxIters and not concentration) or (
            numSamplesList and maxCats and concentration and maxIters and not numSamples) or (
            maxCatsList and concentration and numSamples and maxIters and not maxCats
    ) or (maxItersList and concentration and numSamples and maxCats), "Must define a value for every hyperparameter but one, which should have a list of values to try"


    #Set up the trials to iterate over, based on what parameter has a list given
    if concentration_trials:
        trials = concentration_trials
        title_col = "Concentration"
    elif maxCatsList:
        trials = maxCatsList
        title_col = "Max Categories"
    elif maxItersList:
        trials = maxItersList
        title_col = "Max Iterations"
        maxIters = None #Default maxIters is 100 instead of None like the other params (for backwards compatibility),
        # so set it to None here
    else:
        trials = numSamplesList
        title_col = "NumSamples"

    #If creating learners file, clear out old runs
    if view_learners:
        learner_fn = outputFile[:-4]  # Get filename without .csv
        learner_fn = learner_fn + "_learners.csv"
        if pathlib.Path(learner_fn).exists(): #Clear out file if it exists to append to empty file
            os.remove(learner_fn)



    run_data = defaultdict(lambda: {})
    for index, trial in enumerate(trials):
        if index % 10 == 0: #Output progress for larger runs
            print("Working on", title_col, "trial", trial)

        learners = run_across_seeds(inputLanguage=inputLanguage,
                                    concentration=concentration if concentration else trial,
                                    maxCats=maxCats if maxCats else trial,
                                    numSamples=numSamples if numSamples else trial,
                                    max_iters=maxIters if maxIters else trial,
                                    numSeeds=numSeeds)

        if view_learners:

            plot_dir = "parameter_search_outputs/learner_plots/"+"_".join([str(inputLanguage.name),
                                str(concentration) if concentration else "concentrations",
                                 str(maxCats) if maxCats else "maxCats",
                                 str(numSamples) if numSamples else "numSampless",
                                 str(maxIters if maxIters else "maxIterss")])
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            output_learners(learner_fn,learners, seeds_to_plot=seeds_to_view, plot_dir=plot_dir)


        # concentration_results[trial]
        # concentration_aris[trial] = np.mean([learner.evaluate_accuracy() for learner in learners])
        aris = [learner.evaluate_accuracy() for learner in learners]

        converged = [1 if learner.dpgmm.converged_ else 0 for learner in learners]
        # if debug:
        #     learners[0].plot_predictions()
        #     print(sorted([weight for weight in learners[0].dpgmm.weights_]))
        #     num_cats = [len(learner.learnedLanguage.vowels) for learner in
        #                 learners]  # TODO: This will always be the max number! Need to look at priors on each cat
        #     #exit()

        effective_cat_counts = [len(learner.effective_categories()) for learner in learners]
        height_ARIs = [learner.evaluate_height_categories_ari() for learner in learners]
        nasal_split_counts = [nasal_splits(learner.effective_categories()) for learner in learners]
        run_data[trial] = {
            "Mean ARI": np.mean(aris),
            "Mean height ARI": np.mean(height_ARIs),
            "ARI standard dev": np.std(aris),
            "Percent converged": sum(converged) / len(converged),
            "Mean number of categories": np.mean(effective_cat_counts),
            "Number of categories standard dev": np.std(effective_cat_counts),
            "MaxIters": maxIters,
            "Num seeds": len(learners),
            "Mean nasal splits": np.mean(nasal_split_counts),
            "Stdev nasal splits": np.std(nasal_split_counts)

        }
        if numSamples:
            run_data[trial].update({"Samples":numSamples})
        if maxCats:
            run_data[trial].update({"MaxCats":maxCats})
        if concentration:
            run_data[trial].update({"Concentration":concentration})

    output_to_csv(run_data, outputFile, title_col)



def run_concentration_trials(language:Language, seeds_to_view=[], numSamples=250, maxIters=1000):
    """Code for concentration parameter search
    :param language: Language object to use for input. language.name should be defined
    :param seeds_to_view: list of seeds to save plots of learners' predictions
    """
    assert language.name, "Language name property must be defined"
    maxCats=6
    numSeeds = 20

    #concentration_trials = [(1/(6**i)) for i in range(1,10)] # Look at exponentially small concentrations
    #concentration_trials += [1/(6 + i) for i in range(1,10)] # Denser sample of smaller concentrations
    #concentration_trials += [(1/6) * i for i in range(1, 10)] #Denser sample of bigger concentrations
    concentration_trials = [0.00000016, 1/6, 1, 10, 100, 1000, 10000, 100000]
    concentration_trials.sort()
    debug_str = "debug/" if debug else ""
    outfile_name = ("parameter_search_outputs/" +
                    debug_str +
                    language.name +
                    "_" + "_".join([str(maxCats),
                                    str(numSamples),
                                    str(maxIters)])
                    + "_concentration_search.csv")

    param_search(language, outfile_name, concentration_trials=concentration_trials, maxCats=maxCats,
                 numSamples=numSamples, maxIters=maxIters, numSeeds=numSeeds, view_learners=True,
                 seeds_to_view=seeds_to_view)

    # Find bound concentration parameter to recover categories consistently (across 10 random seeds)
    # Go through range around .16(1 / number of max categories, SciKit default) by intervals of 0.05
    #concentration_trials = [.0000000000001] + [((1 / 6) + 0.05 * i) for i in range(-3, 0)] + [((1 / 6) + 0.05 * i) for i
                                                                                           #   in range(0, 30)
    #concentration_trials.sort()
    #outfile_name = "parameter_search_outputs/"+language.name+"_"+"_".join([str(maxIters), str(maxCats), str(numSamples)])+"_concentration_search.csv"
    #param_search(language, outfile_name, concentration_trials=concentration_trials, maxCats=maxCats, numSamples=numSamples, maxIters =maxIters)
    # concentration_trials = [.0000000000001] + [1 / 6 ** i for i in range(0, 10)] + [
    #     100]  # + [((1/6) + 0.05*i) for i in range(0,15)]


def run_numsamples_trials(language:Language):
    """Code for number of samples  parameter search
        :param language: Language object to use for input. language.name should be defined"""
    assert language.name, "Language name property must be defined"

    numsample_trials = [i for i in range (50, 500, 50)]
    param_search(language,
                 "parameter_search_outputs/"+language.name+"_maxiters_200_numsample_search.csv",
                 numSamplesList=numsample_trials,
                 concentration=1/6,
                 maxCats=6,maxIters=200)

    # Increase max number of iterations so bigger input sample sizes can converge
    param_search(language,
                 "parameter_search_outputs/"+language.name+"_maxiters_1000_numsample_search.csv",
                 numSamplesList=numsample_trials,
                 concentration=1 / 6,
                 maxCats=6, maxIters=1000)


def run_maxiters_trials(language:Language):
    """Code for maximum number of iterations parameter search
        :param language: Language object to use for input. language.name should be defined"""
    assert language.name, "Language name property must be defined"
    numiters_trials = [i for i in range(100, 1000, 100)]
    param_search(language,
                 "parameter_search_outputs/"+language.name+"_numsamples_250_maxiters_search.csv",
                 maxItersList=numiters_trials,
                 concentration=1 / 6,
                 maxCats=6,
                 numSamples=250)


def ideal_model(language, seed, numSamples):
    #Given a language, sampling seed, and number of samples, computes the ARI for the "ideal" case where the
    #language categories are known
    # TODO: make this flexible with cat frequency property. Right now, since assuming even mixing proportions, can just use likelihood

    #Get sample from language
    data, labels = language.sample(seed=seed, num_samples_per_cat=numSamples, get_labels=True)

    #Get pdf for each category in 'true' language
    cats = [multivariate_normal(mean=cat.mean, cov = cat.cov) for cat in language.vowels]

    # Get index of category that has the highest likelihood for each sample
    max_likelihood_labels = [max(*[(index, cat.pdf(sample)) for index,cat in enumerate(cats)], key = lambda item: item[1]) for sample in data]
    predicted = [max_likelihood_label[0] for max_likelihood_label in max_likelihood_labels]
    if debug:
        print(max_likelihood_labels)

    # Compute ARI for 'ideal' predicted categories
    return adjusted_rand_score(labels_true=labels, labels_pred=predicted)


def view_seed_samples(language, seeds, output_dir=None):
    """
    Shows a plot for each seed in seeds with the sample of language using that(with informative title)
    :param language: Language object, language to sample from
    :param seeds: list of seeds (int)
    :param output_fn: if not None, the string name of the directory to output plots to (each named after language name and seed)
    :return: None
    """
    for seed in seeds:
        title = " ".join([language.name if language.name else "", str(seed)])
        if output_dir:
            fn = output_dir + "/" + title.replace(" ", "_")
        else:
            fn = None
        language.plot_categories(showSamples=True,
                                 seed=1,
                                 title=title,
                                 savefilename=fn)

if __name__ == "__main__":

    easy_language = define_easy_language()
    overlap_language = define_slight_overlap_language()
    challenge_language = define_challenge_language()
    scikit_lang = define_scikit_example_language()
    challenge_lang_2 = define_challenge_language_2()
    challenge_lang_2.plot_categories(showSamples=True, seed=9)
    challenge_sphere = define_challenge_sphere()
    challenge_sphere.plot_categories(showSamples=True)

    if debug:
        easy_language.plot_categories(showSamples=True)
        overlap_language.plot_categories(showSamples=True)
        challenge_language.plot_categories(showSamples=True)
        debug_lang = define_debug_language()
        debug_lang.plot_categories(showSamples=True)
        run_concentration_trials(debug_lang)

        print(ideal_model(overlap_language, seed=17, numSamples=250))
        print(ideal_model(challenge_language, seed=17, numSamples=250))
        print(ideal_model(challenge_lang_2, seed=17, numSamples=250))

        #Too big number => impossible to find right number cats?
        #param_search(inputLanguage = overlap_language,
        #outputFile = "parameter_search_outputs/debug/50_max_cat.csv",
        #concentration_trials = [1/(50**2), 1/50, 1/25],
        #numSamples= 300,
        #maxIters = 1000, maxCats=50)
    else:

        #run_concentration_trials(overlap_language)

        #Overlap language: look at seeds 17, 1 (that's where concs. are learning diff number cats)
        #view_seed_samples(overlap_language, [17, 1])
        #run_concentration_trials(overlap_language, seeds_to_view=[17,1,0])

        #Challenge language: look at seeds 17, 18
        #run_concentration_trials(challenge_language, seeds_to_view=[17,18,0])

        for num_samples in [250, 300, 350, 400, 450, 500]:
            run_concentration_trials(challenge_lang_2, seeds_to_view=[0,1,17,18,9], numSamples=num_samples, maxIters=3000)

        for num_samples in [250, 300, 350, 400, 450, 500]:
            run_concentration_trials(challenge_sphere, seeds_to_view=[0,1,17,18,9], numSamples=num_samples, maxIters=3000)

        #print(ideal_model(overlap_language,1,250))
        # scikit_lang.plot_categories(showSamples=True)
        # print("Trying different concentration values...")
        # run_concentration_trials(scikit_lang)
        #
        # print("Trying different concentration values...")
        # run_concentration_trials(overlap_language)



        #print("Trying different numbers of samples...")
        #run_numsamples_trials(overlap_language)

        #print("Trying different numbers of optimization maximum iterations...")
        #run_maxiters_trials(overlap_language)


    #easy_language.plot_categories(showSamples=True)
    #run_concentration_trials(easy_language)
    # run_numsamples_trials(easy_language)
    # run_maxiters_trials(easy_language)

    #
    #overlap_language.plot_categories(showSamples=True)
    # if debug:
    #     overlap_language.plot_categories()
    #   concentration_trials = [.0000000000001] + [((1 / 6) + 0.05 * i) for i in range(-3, 0)] + [((1 / 6) + 0.05 * i) for i
    #                                                                                          in range(0, 30)]
    #   param_search(overlap_language, "parameter_search_outputs/concentration_search_debug.csv",
    #             concentration_trials=concentration_trials, maxCats=6, numSamples=100)




    #run_numsamples_trials(overlap_language)
    #run_maxiters_trials(overlap_language)







