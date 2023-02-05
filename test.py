import pandas as pd
import utils
import numpy as np
import SGD
import computeFunctions as cf
import utils
import seaborn as sns
import time

#define the parameters
alpha_zero = 1
c1 = 1e-4
beta = 0.5
epsilon = 1e-5

def initialize_matrices(n, m, density):
    matrices = []
    starting_points = []
    for i in range(10):
        matrices.append(utils.randomMatrix(n,m,density))
        starting_points.append(np.random.rand(m))
    return [matrices, starting_points]

def run_test(matrices, starting_points, function):
    lists_gradients = []
    lists_residuals = []
    computations = []
    lists_alphas = []
    time_elapsed = []
    time_numpy = []
    for i in range(len(matrices)):
        for j in range(len(starting_points)):
            A = matrices[i]
            x = starting_points[j]
            if function:
                res = SGD.SGD(cf.computeNewFunction, cf.computeGradientNewFunction, x, alpha_zero, beta, c1, epsilon, A, 0)
            else:
                res = SGD.SGD(cf.computeRayleigh, cf.computeGradientRayleigh, x, alpha_zero, beta, c1, epsilon, A, 0)
            lists_gradients.append(res[0])
            computations.append(res[2])
            time_elapsed.append(res[4])
            lists_residuals.append(res[5])
            lists_alphas.append(res[1])
    return [lists_gradients, computations, time_elapsed, lists_residuals, lists_alphas]

def numpy_test(matrices):
    time_numpy = []
    for matrix in matrices:
        start = time.time()
        norm = np.linalg.norm(matrix, ord=2)
        end = time.time()
        time_numpy.append(end - start)
    return time_numpy

def compute_results(lists_gradients, computations, time_elapsed, lists_residuals):
    print('average gradient norm: ', np.mean([l[len(l) - 1] for l in lists_gradients]))
    print('standard deviation gradient norm: ', np.std([l[len(l) - 1] for l in lists_gradients]))
    print('average number of computations:', np.mean(computations))
    print('standard deviation number of computations: ', np.std(computations))
    print('average number of iterations: ', np.mean([len(l) for l in lists_gradients]))
    print('standard deviation number of iterations: ', np.std([len(l) for l in lists_gradients]))
    print('average time: ', np.mean(time_elapsed))
    print('standard deviation time: ', np.std(time_elapsed))
    print('average residual: ', np.mean([l[len(l) - 1] for l in lists_residuals]))
    print('standard deviation residuals: ', np.std([l[len(l) - 1] for l in lists_residuals]))


def plot_alphas(lists_alphas):
    lists_alphas = utils.ResultsMatrix(lists_alphas)
    alphasDF = pd.DataFrame(lists_alphas).melt()
    chart = sns.lineplot(data=alphasDF, x='variable', y='value')
    chart.set(yscale='log', title='alpha values computed by backtracking', xlabel='iterations', ylabel='alpha')


def plot_results(new_function, rayleigh, title):
    new_function = utils.ResultsMatrix(new_function)
    new_functionDF = pd.DataFrame(new_function).melt()
    new_functionDF['type'] = 'new'
    rayleigh = utils.ResultsMatrix(rayleigh)
    rayleighDF = pd.DataFrame(rayleigh).melt()
    rayleighDF['type'] = 'rayleigh'
    DF = pd.concat([rayleighDF,new_functionDF], ignore_index=True)
    chart = sns.lineplot(data=DF, y='value', x='variable', hue='type')
    chart.set(yscale='log', title=title, xlabel='iterations', ylabel='value')