from learner_definitions import Category, Language, Learner
from dirichlet_hyperparam_search import output_to_csv
from mnmc_sim import define_ideal_start
import matplotlib.pyplot as plt
import dirichlet_hyperparam_search
import os


if __name__ == "__main__":
    coartic_strength = .5
    max_cats = 6
    concen = 1.6e-07
    max_iters = 3000
    numSamples = 300
    #bias = .15
    bias = 1.2
    threshold = 1.75
    numGens = 4
    scale = True
    #seeds = list(range(20))

    #Define categories for generating the input distribution, naming the nasal coarticulation categories "nasal"
    input = define_ideal_start(coartic_strength=coartic_strength, name_nas=True)
    input.plot_categories(showSamples=True)

    seed = 5
    #Initalize learner to have context-based bias
    learner0 = Learner(inputLanguage=input,
                           maxCats=max_cats,
                           concentration=concen, max_iters=max_iters,
                           numSamples=numSamples,
                           name="0",
                           contextless_bias=bias,
                           threshold=threshold, seed=seed,
                           context_bias_categories=["nasal"]) #Add category shift names to learner

    #learner0.plot_predictions(title="Learner 0")

    currentLanguage = learner0.learnedLanguage
    currentLanguage.plot_categories(showSamples=True)


