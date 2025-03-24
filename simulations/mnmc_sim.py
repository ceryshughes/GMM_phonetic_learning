from learner_definitions import Category, Language, Learner
from dirichlet_hyperparam_search import output_to_csv
import matplotlib.pyplot as plt
import dirichlet_hyperparam_search
import os
#from dirichlet_hyperparam_search import nasal_splits

def define_ideal_start(coartic_strength=.2):
    """
        Sets up a simple baseline language with 2 height categories, both with a slight sub-category shift for nasality
        :param coartic_strength difference between nasal and oral contexts in nasality
        :return: a Language with 4 categories, named "ideal start"
        """
    high_or = Category.build_params(mean_nasality=1, mean_height=5, s_nasality=.75, s_height=.75, c_nasality_height=-0)
    high_nas = Category.build_params(mean_nasality=1 + coartic_strength, mean_height=5, s_nasality=.75, s_height=.75, c_nasality_height=-0)
    low_or = Category.build_params(mean_nasality=1, mean_height=0, s_nasality=.75, s_height=.75, c_nasality_height=0)
    low_nas = Category.build_params(mean_nasality=1 + coartic_strength, mean_height=0, s_nasality=.75, s_height=.75, c_nasality_height=0)
    return Language(vowels=[high_or, high_nas, low_or, low_nas], name="ideal_start")

def plot_nas_generations(learners:list, savefilename:str = None):
    """
    Creates a line plot of the mean nasality of the two largest categories for each learner, with a label for the
    higher and lower category. Assumes that the learned languages have two large categories, separable based on height.
    :param learners: List of Learner objects, in chronological order of generation
    :param savefilename: optional, name of file to save the plot to
    :return: None
    """
    high_nasalities = []
    low_nasalities = []
    for learner in learners:
        language = learner.learnedLanguage

        #Identify two largest categories
        sorted_cats = sorted([(index, language.priors[index], cat) for index, cat in enumerate(language.vowels)], key=lambda tup: tup[1], reverse=True)
        top_two = sorted_cats[0:2]

        #Identify high and low category
        high_height = max(top_two, key=lambda tup: tup[2].mean[1])
        low_height = min(top_two, key=lambda tup: tup[2].mean[1])

        #Add nasalities to lists for plotting
        high_nasalities.append(high_height[2].mean[0])
        low_nasalities.append(low_height[2].mean[0])

    #Plot nasality of categories over time
    x = list(range(len(learners)))
    plt.plot(x, high_nasalities, label="high", linestyle="--")
    plt.plot(x, low_nasalities, label="low")
    if savefilename:
        plt.savefig(savefilename)
    #plt.show()
    plt.close()


def plot_nasal_split_generations(learner_lists:list, savefilename:str=None):
    """
    For a list of learners (across one or more random seeds),
     plot how many learners have categories have split based on nasality for each generation
    :param learner_lists: List of list of learner: each list of learners corresponds to a random seed, where each learner
        shares that random seed, and are ordered by generation. E.g. [seed1: [learner1, learner2, learner3], seed2: [learner1, learner2, learner3].
        All lists must have the same length.
    :param savefilename: optional, the name of the file to save the plot to
    :return: None
    """
    #Get nasal splits for each generation, for each seed
    nasal_splits = [[dirichlet_hyperparam_search.nasal_splits(learner.learnedLanguage.vowels) for learner in list] for list in learner_lists]
    high_splits = [[nasal_split[1] for nasal_split in list] for list in nasal_splits]
    low_splits = [[nasal_split[2] for nasal_split in list] for list in nasal_splits]

    #Sum nasal splits for each generation across seeds
    sum_across_seeds_high = []
    sum_across_seeds_low = []
    for i in range(len(high_splits[0])):
        high_split_sum = sum([high_splits[list_index][i] for list_index in range(len(high_splits))])
        sum_across_seeds_high.append(high_split_sum)

        low_split_sum = sum([low_splits[list_index][i] for list_index in range(len(low_splits))])
        sum_across_seeds_low.append(low_split_sum)

    #Plot number of nasal splits for each generation
    x = list(range(len(sum_across_seeds_high)))
    plt.plot(x,sum_across_seeds_high, label="high", linestyle="--")
    plt.plot(x, sum_across_seeds_low, label="low")
    if savefilename:
        plt.savefig(savefilename)
    #plt.show()
    plt.close()




if __name__ == "__main__":
    coartic_strength = .5
    input = define_ideal_start(coartic_strength=coartic_strength)
    max_cats = 6
    concen = 1.6e-07
    max_iters = 3000
    numSamples = 300
    bias = .15
    threshold = 1.75
    numGens = 4
    scale = True
    seeds = list(range(20))

    input.plot_categories()
    seed_learners = []
    for seed in seeds:
        learner0 = Learner(inputLanguage=input,
                           maxCats=max_cats,
                           concentration=concen, max_iters=max_iters,
                           numSamples=numSamples,
                           name="0",
                           contextless_bias=bias,
                           threshold=threshold, seed=seed)
        #learner0.plot_predictions(title="Learner 0")

        currentLanguage = learner0.learnedLanguage
        #currentLanguage.plot_categories(showSamples=True)
        learners = []
        for gen in range(0, numGens):
            learner = Learner(inputLanguage=currentLanguage, scale=True,
                              maxCats=max_cats,
                              concentration=concen, max_iters=max_iters,
                              numSamples=numSamples, name=str(gen),
                              contextless_bias=bias,
                              threshold=threshold, seed=seed)
            #print(str(learner))
            currentLanguage = learner.learnedLanguage
            if dirichlet_hyperparam_search.nasal_splits(learner.learnedLanguage.vowels)[0] > 0:
                dir = "mnmc_plots/" + "_".join([str(value) for value in [concen, max_iters, max_cats, numSamples, scale]]) + "/"
                if not os.path.exists(dir):
                    os.mkdir(dir)
                name = "seed_" + str(seed) + "_gen_" + str(gen) + "_" + "converged" if learner.dpgmm.converged_ else "not_converged" + "_predictions"
                learner.plot_predictions(title=name,
                                         savefilename= dir + name + ".png")
            #learner.learnedLanguage.plot_categories(showSamples=True, title=str(gen)+ " categories")
            #learner.plot_predictions(title="Learner "+str(gen))
            learners.append(learner)
        plot_nas_generations(learners)
        seed_learners.append(learners)
    plot_nasal_split_generations(seed_learners, savefilename="mnmc_plots/" +
                                                             "_".join([str(value) for value in [concen, max_iters, max_cats, numSamples, scale]])+
                                                             "/splits_over_time.png")

    #TODO: plot for counting nasal splits





