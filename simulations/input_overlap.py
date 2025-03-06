# Python file for investigating how the number of categories learned by a Dirichlet prior Mixture of Gaussians learner
# varies with the amount of overlap between categories in generating the input data
from learner_definitions import Category, Language, Learner
import numpy as np
import csv
from collections import defaultdict
from dirichlet_hyperparam_search import run_across_seeds, output_to_csv

debug = True


def input_comparison(languages:list,
                 outputFile:str,
                 concentration:float,
                 maxCats:int,
                 numSamples:int,
                 maxIters:int):
    """
        -Runs Learners on each input language provided in languages
        -Tries 10 random seeds for each language
        -Outputs the results (ARIs, convergence, number of effective categories for each language) to a csv file.
        :param languages: list of Language, input languages to try learners on and compare. Each Language should have a self.name property
        :param outputFile: name of the csv output file to put the results in
        :param concentration: the concentration value to use for learners
        :param maxCats the maximum category value to use for learners
        :param numSamples: the number of samples to use per category for input to learners
        :param maxIters: maximum number of learning iterations for each learner
    """


    run_data = defaultdict(lambda: {})
    for inputLanguage in languages:

        learners = run_across_seeds(numSeeds=100,
                                    inputLanguage=inputLanguage,
                                    concentration=concentration,
                                    maxCats=maxCats,
                                    numSamples=numSamples,
                                    max_iters=maxIters)


        # concentration_results[trial]
        # concentration_aris[trial] = np.mean([learner.evaluate_accuracy() for learner in learners])
        aris = [learner.evaluate_accuracy() for learner in learners]

        converged = [1 if learner.dpgmm.converged_ else 0 for learner in learners]
        # if debug:
        #     learners[0].plot_predictions()
        #     print(sorted([weight for weight in learners[0].dpgmm.weights_]))
        #     num_cats = [len(learner.learnedLanguage.vowels) for learner in
        #                 learners]  # TODO: This will always be the max number! Need to look at priors on each cat
        #     exit()

        effective_cat_counts = [len(learner.effective_categories()) for learner in learners]
        run_data[inputLanguage.name] = {
            "Mean ARI": np.mean(aris),
            "ARI standard dev": np.std(aris),
            "Percent converged": sum(converged) / len(converged),
            "Mean number of categories": np.mean(effective_cat_counts),
            "Number of categories standard dev": np.std(effective_cat_counts)
        }
    output_to_csv(run_data, outputFile,  "Language")


def define_language_overlaps():
    height = 1
    cat1_nas = 1
    distances = [0.25 * d for d in range(0, 12)]
    nas_var = .75
    height_var = .75
    nas_height_covar = .1

    languages = []
    cat1 = Category.build_params(mean_nasality=cat1_nas,
                                 mean_height=height,
                                 s_nasality=nas_var,
                                 s_height=height_var,
                                 c_nasality_height=nas_height_covar)
    for index, d in enumerate(distances):
        cat2 = Category.build_params(mean_nasality=cat1_nas+d,
                                     mean_height=height,
                                     s_nasality=nas_var,
                                     s_height=height_var,
                                     c_nasality_height=nas_height_covar
                                     )
        languages.append(Language(vowels=[cat1,cat2], name=str(d)))
    return languages


if __name__ == "__main__":

    languages = define_language_overlaps()
    input_comparison(languages=languages,
                     outputFile="overlap_simulation_outputs/"+"initial_search_overlaps_c=0.016_cats=6_iters=1000_samples=250.csv",
                     concentration=1000,
                     maxCats=6,
                     maxIters=1000,
                     numSamples=250)



