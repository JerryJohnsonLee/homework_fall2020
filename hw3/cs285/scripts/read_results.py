import glob
import tensorflow as tf
import pandas as pd
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def get_results_dataframe(file):
    rtn_dict = {}
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in rtn_dict:
                rtn_dict[v.tag].append(v.simple_value)
            else:
                rtn_dict[v.tag] = [v.simple_value]
    max_len = max([len(arr) for arr in rtn_dict.values()])
    for category in rtn_dict:
        rtn_dict[category] += [np.nan] * (max_len - len(rtn_dict[category]))
    df = pd.DataFrame.from_dict(rtn_dict)
    return df

if __name__ == '__main__':
    import glob

    logdir = '../data/q1_MsPacman-v0_07-10-2020_15-00-56/events*'
    eventfile = glob.glob(logdir)[0]

    df = get_results_dataframe(eventfile)
    X, Y = get_section_results(eventfile)

    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))