# Python file for a hyperparameter search on SciKit Learn's Bayesian Mixture of Gaussians model
from learner_definitions import Category, Language, Learner
import numpy as np
import csv
from collections import defaultdict

debug = False

def run_across_seeds(seeds:list,
                     inputLanguage:Language,
                     numSamples:int = 100,
                     concentration:float = None,
                     maxCats:int = 6,
                     max_iters:int = 100):
    """
    Runs Learners with identical parameters, just with different random seeds, specified in the "seeds" parameter
    :param seeds: list of random seeds (ints) to try
    :param inputLanguage: Input Language for the learner
    :param numSamples:  number of samples per category given to the learner
    :param concentration: Dirichlet prior concentration parameter
    :param maxCats: maximum number of categories available to the learner
    :param max_inters: maximum number of iterations learner can run
    :return: A list of Learner objects, one for each seed in seeds (in order)
    """
    learners = []
    for seed in seeds:
        learner = Learner(seed=seed,
                          inputLanguage=inputLanguage,
                          numSamples=numSamples,
                          concentration=concentration,
                          maxCats=maxCats,
                          max_iters=max_iters)
        learners.append(learner)
    return learners


# Define easy language: 4 nonoverlapping categories
def define_easy_language():
    """
    Sets up a simple baseline language that should be trivial for a cluster learner
    :return: a Language with 4 separable categories
    """
    cat_1 = Category.build_params(mean_nasality=.5, mean_height=.5, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_2 = Category.build_params(mean_nasality=.5, mean_height=8, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_3 = Category.build_params(mean_nasality=8, mean_height=.5, s_nasality=.5, s_height=.5, c_nasality_height=0)
    cat_4 = Category.build_params(mean_nasality=8, mean_height=8, s_nasality=.5, s_height=.5, c_nasality_height=0)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4])

def define_slight_overlap_language():
    """
        Sets up a simple baseline language that should be easy for a cluster learner
        :return: a Language with 4 separable categories
        """
    cat_1 = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_2 = Category.build_params(mean_nasality=1, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_3 = Category.build_params(mean_nasality=4, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    cat_4 = Category.build_params(mean_nasality=4, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=.1)
    return Language(vowels=[cat_1, cat_2, cat_3, cat_4])


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
    file = open(filename, "w+")
    csv_writer = csv.DictWriter(file, fieldnames=[run_col]+list(list(runs.values())[0].keys()))
    csv_writer.writeheader()
    for setting in runs:
        row = runs[setting]
        row.update({run_col:setting})
        csv_writer.writerow(row)

    file.close()
    return


def concentration_trial_param_search(inputLanguage:Language, outputFile:str, concentration_trials:list, maxCats:int, numSamples:int):
    if debug:
        inputLanguage.plot_categories()

    seeds = [i for i in range(0,10)]
    concentration_run_data = defaultdict(lambda: {})
    for trial in concentration_trials:
        learners = run_across_seeds(seeds,
                                    inputLanguage=inputLanguage,
                                    concentration=trial, maxCats=maxCats, numSamples=numSamples)
        # concentration_results[trial]
        # concentration_aris[trial] = np.mean([learner.evaluate_accuracy() for learner in learners])
        aris = [learner.evaluate_accuracy() for learner in learners]

        converged = [1 if learner.dpgmm.converged_ else 0 for learner in learners]
        if debug:
            learners[0].plot_predictions()
            print(sorted([weight for weight in learners[0].dpgmm.weights_]))
            num_cats = [len(learner.learnedLanguage.vowels) for learner in
                        learners]  # TODO: This will always be the max number! Need to look at priors on each cat
            exit()

        effective_cat_counts = [len(learner.effective_categories()) for learner in learners]
        concentration_run_data[trial] = {
            "Mean ARI": np.mean(aris),
            "ARI standard dev": np.std(aris),
            "Percent converged": sum(converged) / len(converged),
            "Mean number of categories": np.mean(effective_cat_counts),
            "Number of categories standard dev": np.std(effective_cat_counts)
        }
    output_to_csv(concentration_run_data, outputFile, "Concentration")


def param_search(inputLanguage:Language,
                 outputFile:str,
                 concentration_trials:list = None,
                 concentration:float = None,
                 maxCatsList: list = None,
                 maxCats:int = None,
                 numSamples:int = None,
                 numSamplesList: list = None,
                 maxIters:int = 100):
    assert (concentration_trials and maxCats and numSamples and not concentration) or (
            numSamplesList and maxCats and concentration and not numSamples) or (
            maxCatsList and concentration and numSamples and not maxCats
    ), "Must define a value for every hyperparameter but one, which should have a list of values to try"

    if debug:
        inputLanguage.plot_categories()

    if concentration_trials:
        trials = concentration_trials
        title_col = "Concentration"
    elif maxCatsList:
        trials = maxCatsList
        title_col = "MaxCats"
    else:
        trials = numSamplesList
        title_col = "NumSamples"


    seeds = [i for i in range(0,10)]
    concentration_run_data = defaultdict(lambda: {})
    for trial in trials:

        learners = run_across_seeds(seeds,
                                    inputLanguage=inputLanguage,
                                    concentration=concentration if concentration else trial,
                                    maxCats=maxCats if maxCats else trial,
                                    numSamples=numSamples if numSamples else trial,
                                    max_iters=maxIters)


        # concentration_results[trial]
        # concentration_aris[trial] = np.mean([learner.evaluate_accuracy() for learner in learners])
        aris = [learner.evaluate_accuracy() for learner in learners]

        converged = [1 if learner.dpgmm.converged_ else 0 for learner in learners]
        if debug:
            learners[0].plot_predictions()
            print(sorted([weight for weight in learners[0].dpgmm.weights_]))
            num_cats = [len(learner.learnedLanguage.vowels) for learner in
                        learners]  # TODO: This will always be the max number! Need to look at priors on each cat
            exit()

        effective_cat_counts = [len(learner.effective_categories()) for learner in learners]
        concentration_run_data[trial] = {
            "Mean ARI": np.mean(aris),
            "ARI standard dev": np.std(aris),
            "Percent converged": sum(converged) / len(converged),
            "Mean number of categories": np.mean(effective_cat_counts),
            "Number of categories standard dev": np.std(effective_cat_counts)
        }
    output_to_csv(concentration_run_data, outputFile,  title_col)


if __name__ == "__main__":
    easy_language = define_slight_overlap_language()

    if debug:
        easy_language.plot_categories()

    # Find bound concentration parameter to recover categories consistently (across 10 random seeds)
    #Go through range around .16 (1/number of max categories, SciKit default) by intervals of 0.05
    #concentration_trials = [.0000000000001] + [((1/6) + 0.05*i) for i in range(-3,0)] + [((1/6) + 0.05*i) for i in range(0,30)]
    #concentration_trial_param_search(easy_language, "parameter_search_outputs/concentration_search.csv", concentration_trials, maxCats=6, numSamples=50)
    #concentration_trials = [.0000000000001] + [1/6**i for i in range(0,10)] + [100] #+ [((1/6) + 0.05*i) for i in range(0,15)]
    #concentration_results = dict()


    # Vary number of samples (across 10 random seeds)
    numsample_trials = [i for i in range (50, 500, 50)]
    param_search(easy_language,
                 "parameter_search_outputs/numsample_search.csv",
                 numSamplesList=numsample_trials,
                 concentration=1/6,
                 maxCats=6)
    #Increase max number of iterations so bigger input sample sizes can converge
    param_search(easy_language,
                 "parameter_search_outputs/maxiters_200_numsample_search.csv",
                 numSamplesList=numsample_trials,
                 concentration=1/6,
                 maxCats=6,maxIters=200)




