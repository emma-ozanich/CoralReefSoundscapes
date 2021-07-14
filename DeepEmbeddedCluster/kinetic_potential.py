def kinetic_potential(Sx1, Sx2, Sx3): # starting for gsi file

    import numpy as np
    
    # check for input dimensionality
    if (Sx1.shape[0]>Sx1.shape[1]) or (Sx2.shape[0]>Sx2.shape[1]) or (Sx3.shape[0]>Sx3.shape[1]):
        print('Please ensure the channel dimension is first, followed by sample dimension.')
        return
    
    KEtoPE = (np.abs(Sx2)**2 + np.abs(Sx3)**2) / (np.abs(Sx1)**2) # velocity autospectrum/ pressure autospectrum   
    
    return KEtoPE
