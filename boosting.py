# Boosting procedure
# Alec Myres
# MATH G4073 - Columbia University

import numpy as np
import pandas as pd

# df is data frame of observations, with last row corresponding
# to "current" time period for reweighting
#
# Columns:
# "Date"
# "Price"
# "y" (outcome, {1,-1})
# "g_1", "g_2", etc. (weak learners {1,-1}, from indicators) 
def update_weights(df, alphas):
    # get row index for final row in data frame
    r = df.index[-1]
    n = len(df.index)

    # get column names of learners
    learners = [i for i in df.columns if 'g_' in i]
    M = len(learners)
    sortedLearners = []
    alphas = []

    # initialize equal weights
    weights = [1.0/n]*n

    for m in range(M):
        # get weighted errors for each learner
        errors = []
        for l in learners:
            errors.append(np.dot(weights, np.where(df['y'] != df[l], 1, 0))/sum(weights))

        # minimum error and alpha
        minErr, minIndex = min((val, idx) for (idx, val) in enumerate(errors))
        minLearner = learners[minIndex]
        alpha = np.log((1 - minErr)/minErr)
        alphas.append(alpha)

        # update weight vector
        errVect = np.where(df['y'] != df[minLearner], 1, 0)
        weights = np.dot(weights, np.exp(np.multiply(alpha, errVect)))

        # exclude learner
        sortedLearners.append(minLearner)
        learners.remove(minLearner)

    # ADT
    sortedLearnerVals = map(lambda x: df[x][-1], sortedLearners)
    ADT = np.sign(np.dot(alphas, sortedLearnerVals))

    return ADT
        
