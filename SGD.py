import utils
import computeFunctions as cf
import LineSearch as ls
import numpy as np
import math
import time

def SGD(fun, grad, x_zero, alpha_zero, beta, c1, epsilon, A, verbose):
    '''
    :param fun: function to be used
    :param grad: gradient of the given function
    :param x_zero: startfing point
    :param alpha_zero: initial alpha
    :param beta: shrinking value for backtracking
    :param c1: first parameter of Wolfe condition
    :param epsilon: stopping criterion
    :param A: matrix
    :param verbose: Whether to show or not additional informations
    :return: the point minimizing the function and the number of iterations
    '''
    numpy_norm = np.linalg.norm(A, ord=2)
    Q = np.dot(A.T, A)
    start = time.time()
    x = x_zero
    iterations = 0
    list_gradients = []
    list_alphas = []
    list_residuals = []
    g_x = grad(x, Q)
    list_gradients.append(utils.norm(g_x[0]))
    #count the number of computations performed (both gradient and fucntion computations)
    computations = 1

    while utils.norm(g_x[0]) > epsilon and iterations < 100:
        iterations += 1
        res_backtracking = ls.backtracking(fun, grad, x, alpha_zero, beta, c1, Q, g_x ,0)
        alpha = res_backtracking[0]
        computations += res_backtracking[1]
        if verbose:
            print('descent direction: ', np.dot(-g_x[0], g_x[0]))
            print('alpha: ', alpha)
            print('gradient norm: ', utils.norm(g_x[0]))
        x = x - (alpha * g_x[0])
        g_x = grad(x, Q)
        computations += 1
        list_gradients.append(utils.norm(g_x[0]))
        list_alphas.append(alpha)
        list_residuals.append(math.fabs(math.sqrt(-(cf.computeRayleigh(x, Q, g_x))) - numpy_norm))

    end = time.time()
    '''if verbose:
        print('Finished after ', iterations, '')
        print('Norm of the gradient: ', utils.norm(g_x[0]))
        res = math.sqrt(-(cf.computeRayleigh(x, Q, g_x)))
        print('Our result: ', res)
        print('Matrix norm: ', numpy_norm)
        print('total number of computations needed: ', computations)
        print('------------------------------------------------')'''

    return [list_gradients, list_alphas, computations, iterations, (end - start), list_residuals]

