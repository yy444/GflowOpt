from numpy.random import default_rng

from dag_gflownet.scores import BDeScore, BGeScore, priors, BICScore
from dag_gflownet.utils.data import get_data


def get_prior(name):
    prior = {
        'uniform': priors.UniformPrior,
    }
    return prior[name]()


def get_scorer(args, rng=default_rng()):
    # Get the data
    graph, data, score = get_data(args.graph, args, rng=rng)

    # Get the prior
    prior = get_prior(args.prior)
    # Get the scorer
    scores = {'bde': BDeScore, 'bge': BGeScore, 'bic': BICScore}

    #score = 'bic'
    if score != 'bic':
        scorer = scores[score](data, prior)
    else:
        scorer = scores[score](data)

    return scorer, data, graph