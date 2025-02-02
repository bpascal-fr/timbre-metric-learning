import numpy as np



def logpearson(w,f,ld):
    
    """
    Compute the opposite Pearson correlation between parametric distance and human dissimilarity ratings.
    
    Args:
    
        w: weights of the parametric distance, vector of length n
        
        f: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        ld: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        lp: opposite Pearson correlation, real number
    
        
    """
    
    
    # standardize the dissimiarity vector
    ld = (ld - np.mean(ld))/np.std(ld)
    
    # dimension of features and number of sounds
    (D,N) = np.shape(f)
    
    # compute and store the log-kernel
    lk = np.zeros(N*(N-1)//2)
    l = 0
    for i in range(N):
        for j in range(i+1,N):
            lk[l] = -sum(w**(-2)*(f[:,i]-f[:,j])**2)
            l += 1
    lp = 2/(N*(N-1))*np.sum((lk - np.mean(lk))/np.std(lk) * ld)
    
    return lp


def logpearson_nan(w,f,ld):
    
    """
    Compute the opposite Pearson correlation between parametric distance and human dissimilarity ratings.
    
    Args:
    
        w: weights of the parametric distance, vector of length n
        
        f: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        ld: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        lp: opposite Pearson correlation, real number
    
        
    """
    
    
    # standardize the dissimiarity vector
    ld = (ld - np.nanmean(ld))/np.nanstd(ld)
    
    # dimension of features and number of sounds
    (D,N) = np.shape(f)
    
    # compute and store the log-kernel
    lk = np.zeros(N*(N-1)//2)
    l = 0
    for i in range(N):
        for j in range(i+1,N):
            lk[l] = -sum(w**(-2)*(f[:,i]-f[:,j])**2)
            l += 1
    lp = 2/(N*(N-1))*np.nansum((lk - np.nanmean(lk))/np.nanstd(lk) * ld)
    
    return lp






def dlogpearson(w,f,ld):
    
    """
    Compute the gradient with respect to the weights of the opposite Pearson correlation between parametric distance and human dissimilarity ratings
    
    Args:
    
        w: weights of the parametric distance, vector of length n
        
        f: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        ld: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        lp: opposite Pearson correlation, real number
        
        dlp: gradient of the opposite Pearson correlation, vector of length n
        
        dlk: Jacobian of the pairwise Pearson correlations, matrix of size s x n with s = l x (l-1) / 2
    
        
    """
    
    # standardize the dissimiarity vector
    ld = (ld - np.mean(ld))/np.std(ld)
    
    # dimension of features and number of sounds
    (D,N) = np.shape(f)
    
    # derivative of the log-kernel w.r.t. the weights
    lk  = np.zeros(N*(N-1)//2)
    dlk = np.zeros((N*(N-1)//2,D))
    l   = 0
    for i in range(N):
        for j in range(i+1,N):
            lk[l]    = -sum(w**(-2)*(f[:,i]-f[:,j])**2)
            dlk[l,:] = 2*w**(-3)*(f[:,i]-f[:,j])**2
            l += 1
            
    # derivative of the mean of the log-kernel
    mlk  = np.mean(lk)
    dmlk = (N*(N-1)//2)**(-1)*sum(dlk,0);
    
    # derivative of the standard deviation of the log-kernel
    slk  = np.std(lk)
    dslk = slk**(-1)*(N*(N-1)//2)**(-1)*np.dot((lk-mlk).T,dlk-dmlk);
    
    # derivative of the Pearson correlation
    lp  = 2/(N*(N-1))*sum((lk - np.mean(lk))/np.std(lk) * ld)
    dlp = 2/(N*(N-1))*np.dot(slk**(-1)*ld,dlk - dmlk) - 2/(N*(N-1))*sum((lk - mlk)*ld)*dslk/slk**2
    
    return lp, dlp, dlk




def dlogpearson_nan(w,f,ld):
    
    """
    Compute the gradient with respect to the weights of the opposite Pearson correlation between parametric distance and human dissimilarity ratings
    
    Args:
    
        w: weights of the parametric distance, vector of length n
        
        f: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        ld: human dissimilarity ratings stored in a l x (l-1)/2 vector
    
    
    Returns:
    
        lp: opposite Pearson correlation, real number
        
        dlp: gradient of the opposite Pearson correlation, vector of length n
        
        dlk: Jacobian of the pairwise Pearson correlations, matrix of size s x n with s = l x (l-1) / 2
    
        
    """
    
    # standardize the dissimiarity vector
    ld = (ld - np.nanmean(ld))/np.nanstd(ld)
    
    # dimension of features and number of sounds
    (D,N) = np.shape(f)
    
    # derivative of the log-kernel w.r.t. the weights
    lk  = np.zeros(N*(N-1)//2)
    dlk = np.zeros((N*(N-1)//2,D))
    l   = 0
    for i in range(N):
        for j in range(i+1,N):
            lk[l]    = -sum(w**(-2)*(f[:,i]-f[:,j])**2)
            dlk[l,:] = 2*w**(-3)*(f[:,i]-f[:,j])**2
            l += 1
            
    # derivative of the mean of the log-kernel
    mlk  = np.nanmean(lk)
    dmlk = (N*(N-1)//2)**(-1)*np.nansum(dlk,0);
    
    # derivative of the standard deviation of the log-kernel
    slk  = np.nanstd(lk)
    dslk = slk**(-1)*(N*(N-1)//2)**(-1)*np.dot((lk-mlk).T,dlk-dmlk);
    
    # derivative of the Pearson correlation
    lp  = 2/(N*(N-1))*np.nansum((lk - np.nanmean(lk))/np.nanstd(lk) * ld)
    dlp = 2/(N*(N-1))*np.dot(slk**(-1)*ld[~np.isnan(ld)],dlk[~np.isnan(ld),:] - dmlk) - 2/(N*(N-1))*np.nansum((lk - mlk)*ld)*dslk/slk**2
    
    return lp, dlp, dlk

