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
def update_weights(df, side):
    # get row index for final row in data frame
    r = df.index[-1]
    n = len(df.index)
    df['index1'] = range(n)
    n1 = abs(sum(df.query('y_' + side + ' != 0')['y_'+side]))

    # get column names of learners
    learners = [i for i in df.columns if (('action' in i) and (side in i))]
    M = len(learners)
    sortedLearners = []
    alphas = []

    # initialize equal weights
    weights = [1.0/n]*n

    for m in range(M):
        # get weighted errors for each learner
        errors = []

        # Sample 0 values in y column
        sa = np.random.choice(df.query('y_' + side + ' == 0').index, size = n1, replace = False)
        df_sample = pd.concat([df.query('y_' + side + ' != 0'), df.loc[sa]])
        df_sample.sort_index(axis = 0, inplace = True)
        sample_index = df_sample['index1']
        sw = [weights[i] for i in sample_index]

        for l in learners:
            errors.append(np.dot(sw, np.where(df_sample['y_' + side] != df_sample[l], 1, 0))/sum(sw))

        # minimum error and alpha
        minErr, minIndex = min((val, idx) for (idx, val) in enumerate(errors))
        minLearner = learners[minIndex]
        alpha = np.log((1 - minErr)/minErr)
        alphas.append(alpha)

        # update weight vector
        errVect = np.where(df_sample['y_' + side] != df_sample[minLearner], 1, 0)
        sw = np.multiply(sw, np.exp(np.multiply(alpha, errVect)))
        counter = 0
        for s in sample_index:
            weights[s] = sw[counter]
            counter += 1

        # exclude learner
        sortedLearners.append(minLearner)
        learners.remove(minLearner)

    # ADT
    df = df.replace(0, -1)
    sortedLearnerVals = map(lambda x: df[x][r], sortedLearners)
    ADT = np.sign(np.dot(alphas, sortedLearnerVals))

    return alphas, sortedLearners
        
