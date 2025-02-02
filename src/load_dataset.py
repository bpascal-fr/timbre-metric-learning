import pickle            as pkl
import numpy             as np

def load_data(the_dataset,the_representation):
    
    """
    Load human dissimilarity ratings and the audio sample representations.
    
    Args:
    
        the_dataset: string containing the name of the dataset among
        - 'Grey1977'
        - 'Grey1978'
        - 'Iverson1993_Whole'
        - 'Iverson1993_Onset'
        - 'Iverson1993_Remainder'
        - 'McAdams1995'
        - 'Lakatos2000_Harm'
        - 'Lakatos2000_Perc'
        - 'Lakatos2000_Comb'
        - 'Barthet2010'
        - 'Patil2012_A3'
        - 'Patil2012_DX4'
        - 'Patil2012_GD4'
        - 'Siedenburg2016_e2set1'
        - 'Siedenburg2016_e2set2'
        - 'Siedenburg2016_e2set3'
        - 'Siedenburg2016_e3'

        the_representation: string containing the name of the representation among
        - 'strf'
        - 'stft'
        - 'spectrum' (cochlea in the companion paper)
        - 'scattering'
        - 'clap'
        - 'encodec'
        - 'mert'
        - 'mertcat'
    
    
    Returns:
    
        r: n-dimensional representations of the l sounds composing the dataset, matrix of size l x n
        
        D: human dissimilarity ratings stored in a l x l matrix
        
        d: human dissimilarity ratings stored in a l x (l-1)/2 vector
    """

    if type(the_dataset) == list:

        r,D,d = load_multiple_data(the_dataset,the_representation)

    else:
        
        with open('data/'+the_representation+'/'+the_dataset+'.pkl','rb') as Sp:
        
            data = pkl.load(Sp)
            
        # representation
        r = data['representations']
        
        # dissimilarity matrix
        D = data['dissimilarities']
        
        # dissimilarity vector
        P = np.size(D,0)
        d = np.zeros(P*(P-1)//2)
        l = 0
        for i in range(P):
            for j in range(i+1,P):
                d[l] = D[i,j]
                l += 1
    
    return r,D,d


def load_multiple_data(the_datasets,the_representation):
    
    """
    Load human dissimilarity ratings and the audio sample representations.
    
    Args:
    
        the_datasets: list of strings containing the name of the datasets among
        - 'Grey1977'
        - 'Grey1978'
        - 'Iverson1993_Whole'
        - 'Iverson1993_Onset'
        - 'Iverson1993_Remainder'
        - 'McAdams1995'
        - 'Lakatos2000_Harm'
        - 'Lakatos2000_Perc'
        - 'Lakatos2000_Comb'
        - 'Barthet2010'
        - 'Patil2012_A3'
        - 'Patil2012_DX4'
        - 'Patil2012_GD4'
        - 'Siedenburg2016_e2set1'
        - 'Siedenburg2016_e2set2'
        - 'Siedenburg2016_e2set3'
        - 'Siedenburg2016_e3'

        the_representation: string containing the name of the representation among
        - 'strf'
        - 'stft'
        - 'spectrum' (cochlea in the companion paper)
        - 'scattering'
        - 'clap'
        - 'encodec'
        - 'mert'
        - 'mertcat'
    
    
    Returns:
    
        r: n-dimensional representations of the L sounds composing the datasets, matrix of size L x n
        
        D: human dissimilarity ratings stored in a L x L matrix
        
        d: human dissimilarity ratings stored in a L x (L-1)/2 vector
    """

    # load the first dataset for initialization

    the_dataset = the_datasets[0]
    
    with open('data/'+the_representation+'/'+the_dataset+'.pkl','rb') as Sp:
        
            data = pkl.load(Sp)
            
    # representation
    r = data['representations']
    
    # dissimilarity matrix
    D = data['dissimilarities']
    
    # dissimilarity vector
    P = np.size(D,0)
    d = np.zeros(P*(P-1)//2)
    l = 0
    for i in range(P):
        for j in range(i+1,P):
            d[l] = D[i,j]
            l += 1    

    # loop over the multiple datasets

    for the_dataset in the_datasets[1:]:
        
        with open('data/'+the_representation+'/'+the_dataset+'.pkl','rb') as Sp:
        
            data = pkl.load(Sp)
            
        # representation
        r  = np.concatenate((r,data['representations']),axis = 1)
        
        # dissimilarity matrix
        L = np.size(D,0)
        l = np.size(data['dissimilarities'],0)
        M = np.concatenate((np.nan * np.ones((l,L)),data['dissimilarities']),axis = 1)
        D = np.concatenate((D,np.nan * np.ones((L,l))),axis = 1)
        D = np.concatenate((D,M),axis = 0)

        
    # dissimilarity vector
    P = np.size(D,0)
    d = np.zeros(P*(P-1)//2)
    l = 0
    for i in range(P):
        for j in range(i+1,P):
            d[l] = D[i,j]
            l += 1
    
    return r,D,d