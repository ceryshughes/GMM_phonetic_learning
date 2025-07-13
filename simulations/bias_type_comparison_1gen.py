from learner_definitions import Category, Language, Learner
from dirichlet_hyperparam_search import output_learners, nasal_splits
from mnmc_sim import define_ideal_start
import os

debug = True

if __name__ == "__main__":
    coartic_strength = .5
    max_cats = 6
    concen = 1.6e-07
    max_iters = 3000
    numSamples = 300
    #bias = .15
    #bias = 1.2
    bias = 1.8
    height_threshold = 2.5
    min_seed = 1
    max_seed = 30
    output_filename = "bias_type_sims/model_bias_comparison_sims_"+str(coartic_strength)+"_"+str(bias)+".csv"

    #Define generating/input/parent language settings
    input = define_ideal_start(coartic_strength=coartic_strength, name_nas=True)

    if debug:
        input.plot_categories(showSamples=True)

    #Clear out and overwrite output file, if it already exists
    if os.path.exists(output_filename):
        os.remove(output_filename)


    #Repeat with different random seeds
    for seed in range(min_seed, max_seed+1):
        print("Working on seed", str(seed))
        #Run each learner (no bias, context-less, context):
        baseline_learner =  Learner(name = "base", inputLanguage=input, #Label learner with name 'base'
                               maxCats=max_cats,
                               concentration=concen, max_iters=max_iters,
                               numSamples=numSamples,
                               seed=seed)
        contextless_learner =  Learner(name = "contextless", inputLanguage=input, #Label learner with name 'contextless'
                               maxCats=max_cats,
                               concentration=concen, max_iters=max_iters,
                               numSamples=numSamples,
                               contextless_bias=bias, #bias
                               threshold=height_threshold, seed=seed)
        context_learner =  Learner(name = "context", inputLanguage=input, #Label learner with name 'context'
                               maxCats=max_cats,
                               concentration=concen, max_iters=max_iters,
                               numSamples=numSamples,
                               contextless_bias=bias,
                               threshold=height_threshold, seed=seed,
                               context_bias_categories=["nasal"]) #context bias
        # Append learner info and high, low nasal splits to table
        if debug:
            output_learners(output_filename, [baseline_learner, contextless_learner, context_learner],
                            record_names=True, threshold=height_threshold,
                            seeds_to_plot=list(range(min_seed, max_seed+1)),
                            plot_dir="bias_debug_plots", filetype="png")
        else:
            output_learners(output_filename, [baseline_learner, contextless_learner, context_learner], record_names=True, threshold=height_threshold)
