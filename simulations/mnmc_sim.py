from learner_definitions import Category, Language, Learner
from dirichlet_hyperparam_search import output_to_csv


def define_ideal_start():
    """
        Sets up a simple baseline language with 2 height categories, both with a slight sub-category shift for nasality
        :return: a Language with 4 categories, named "ideal start"
        """
    high_or = Category.build_params(mean_nasality=1, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=-.1)
    high_nas = Category.build_params(mean_nasality=1.2, mean_height=4, s_nasality=.75, s_height=.75, c_nasality_height=-.1)
    low_or = Category.build_params(mean_nasality=1, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=-.1)
    low_nas = Category.build_params(mean_nasality=1.2, mean_height=1, s_nasality=.75, s_height=.75, c_nasality_height=-.1)
    return Language(vowels=[high_or, high_nas, low_or, low_nas], name="ideal_start")



if __name__ == "__main__":
    input = define_ideal_start()
    max_cats = 6
    concen = 1.6e-07
    max_iters = 3000
    numSamples = 300
    bias = 0.5
    threshold = 1.75
    numGens = 4

    input.plot_categories()
    learner0 = Learner(inputLanguage=input,
                       maxCats=max_cats,
                       concentration=concen, max_iters=max_iters,
                       numSamples=numSamples,
                       name="0",
                       contextless_bias=bias,
                       threshold=threshold)
    learner0.plot_predictions(title="Learner 0")

    currentLanguage = learner0.learnedLanguage
    for gen in range(0, numGens):
        learner = Learner(inputLanguage=currentLanguage,
                          maxCats=max_cats,
                          concentration=concen, max_iters=max_iters,
                          numSamples=numSamples, name=str(gen),
                          contextless_bias=bias,
                          threshold=threshold)
        print(str(learner))
        currentLanguage = learner.learnedLanguage
        learner.learnedLanguage.plot_categories()
        #learner.plot_predictions(title="Learner "+str(gen))




