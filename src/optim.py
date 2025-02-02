import numpy             as np
import scipy.optimize    as so

from   pearson           import logpearson, dlogpearson, logpearson_nan, dlogpearson_nan
from   loss              import compute_A
    

def bfgs_log_kernel(r,d,w_mean = 1,w_std  = 0.01):
    
    
    """
    Learn the metric weights by maximizing the Pearson correlation between the squared parametric distance and human dissimilarity ratings with i.i.d. Gaussian random initialization following (Thoret, 2021, Nat. Hum. Behav.).
    
    Args:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
        
        w_mean: mean of the initial random weight vector
        
        w_std: standard deviation of the initial random weight vector
    
    
    Returns:
    
        res: structure containing
            
            - res.x: optimal weights, vector of dimension n
            
            - res.fun: opposite maximal Pearson correlation, real number
            
            - res.success: boolean indicating whether the L-BFGS-B algorithm has reached its convergence conditions
    
        
    """

    if np.isnan(np.sum(d)):

        res = bfgs_log_kernel_nan(r,d,w_mean,w_std)
    
    else:
    
        # feature dimension
        M     = np.size(r,0)
        
        # pearson correlation to optimize
        def fun(w):
            lp    = logpearson(w,r,d)
            return lp
        
        # gradient of the pearson correlation
        def jac(w):
            lk,dlk,dt    = dlogpearson(w,r,d)
            return dlk
    
        # define the bounds on weights
        optim_bounds = [(1.0*f,1e15*f) for f in np.ones((M,))]
    
        # initalize the weights at random
        w0     = np.abs(w_mean + w_std*np.random.randn(M))
    
        # settings
        optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': 10000, 'ftol': 1e-36}
        
        # run BFGS algorithm
        res    = so.minimize(fun, method='L-BFGS-B', jac=jac, x0 = w0, options = optim_options, bounds = optim_bounds)
        
        if res.success:
            print('BFGS has converged.')
            
    return res


    
def bfgs_log_kernel_w1(r,d):
    
    
    """
    Learn the metric weights by maximizing the Pearson correlation between the squared parametric distance and human dissimilarity ratings with warm initialization computed from a nonnegative least-square regression.
    
    Args:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        res: structure containing
        
            - res.x: optimal weights, vector of dimension n
            
            - res.fun: opposite maximal Pearson correlation, real number
            
            - res.success: boolean indicating whether the L-BFGS-B algorithm has reached its convergence conditions
        
    """

    if np.isnan(np.sum(d)):

        res = bfgs_log_kernel_w1_nan(r,d,w_mean,w_std)
    
    else:
    
        # feature dimension
        M     = np.size(r,0)
        
        # pearson correlation to optimize
        def fun(w):
            lp    = logpearson(w,r,d)
            return lp
        
        # gradient of the pearson correlation
        def jac(w):
            lk,dlk,dt    = dlogpearson(w,r,d)
            return dlk
    
        # define the bounds on weights
        optim_bounds = [(1.0*f,1e15*f) for f in np.ones((M,))]
    
        # initalize the weights solving non-negative least squares
        A     = compute_A(r)
        x,rsd = so.nnls(A, d, maxiter=None)
        w1    = np.sqrt(1/(x+np.finfo(float).eps))
    
    
        # settings
        optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': 10000, 'ftol': 1e-36}
        
        # run BFGS algorithm
        res    = so.minimize(fun, method='L-BFGS-B', jac=jac, x0 = w1, options = optim_options, bounds = optim_bounds)
        
        if res.success:
            print('BFGS has converged.')
        else:
            print('BFGS has not converged.')
        
    return res



def bfgs_log_kernel_nan(r,d,w_mean = 1,w_std  = 0.01):
    
    
    """
    Learn the metric weights by maximizing the Pearson correlation between the squared parametric distance and human dissimilarity ratings with i.i.d. Gaussian random initialization following (Thoret, 2021, Nat. Hum. Behav.).
    
    Args:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
        
        w_mean: mean of the initial random weight vector
        
        w_std: standard deviation of the initial random weight vector
    
    
    Returns:
    
        res: structure containing
            
            - res.x: optimal weights, vector of dimension n
            
            - res.fun: opposite maximal Pearson correlation, real number
            
            - res.success: boolean indicating whether the L-BFGS-B algorithm has reached its convergence conditions
    
        
    """
    
    # feature dimension
    M     = np.size(r,0)
    
    # pearson correlation to optimize
    def fun(w):
        lp    = logpearson_nan(w,r,d)
        return lp
    
    # gradient of the pearson correlation
    def jac(w):
        lk,dlk,dt    = dlogpearson_nan(w,r,d)
        return dlk

    # define the bounds on weights
    optim_bounds = [(1.0*f,1e15*f) for f in np.ones((M,))]

    # initalize the weights at random
    w0     = np.abs(w_mean + w_std*np.random.randn(M))

    # settings
    optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': 10000, 'ftol': 1e-36}
    
    # run BFGS algorithm
    res    = so.minimize(fun, method='L-BFGS-B', jac=jac, x0 = w0, options = optim_options, bounds = optim_bounds)
    
    if res.success:
        print('BFGS has converged.')
    return res


def bfgs_log_kernel_w1(r,d):
    
    
    """
    Learn the metric weights by maximizing the Pearson correlation between the squared parametric distance and human dissimilarity ratings with warm initialization computed from a nonnegative least-square regression.
    
    Args:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        res: structure containing
        
            - res.x: optimal weights, vector of dimension n
            
            - res.fun: opposite maximal Pearson correlation, real number
            
            - res.success: boolean indicating whether the L-BFGS-B algorithm has reached its convergence conditions
        
    """

    # feature dimension
    M     = np.size(r,0)
    
    # pearson correlation to optimize
    def fun(w):
        lp    = logpearson_nan(w,r,d)
        return lp
    
    # gradient of the pearson correlation
    def jac(w):
        lk,dlk,dt    = dlogpearson_nan(w,r,d)
        return dlk

    # define the bounds on weights
    optim_bounds = [(1.0*f,1e15*f) for f in np.ones((M,))]

    # initalize the weights solving non-negative least squares
    A     = compute_A(r)
    x,rsd = so.nnls(A[~np.isnan(d)], d[~np.isnan(d)], maxiter=None)
    w1    = np.sqrt(1/(x+np.finfo(float).eps))


    # settings
    optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': 10000, 'ftol': 1e-36}
    
    # run BFGS algorithm
    res    = so.minimize(fun, method='L-BFGS-B', jac=jac, x0 = w1, options = optim_options, bounds = optim_bounds)
    
    if res.success:
        print('BFGS has converged.')
    else:
        print('BFGS has not converged.')
        
    return res

    
