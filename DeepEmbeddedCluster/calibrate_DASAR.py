def calibrate_DASAR(x,dname):
    
    import numpy as np
    import scipy.signal as sp
    
    # define the filter parameters based on instrument type
    a_filt = np.array((1, -2.911197067426073, 2.826172902227507, -0.9149758348014339))
    b_filt = np.array((0.5140662826979191, -0.9510226229911504, 0.3598463978885433, 0.07710994240468787))
    if dname=='DASARC' or dname=='NorthStar08':
        amp_scale = (2.5/65535)*(10**(149/20))
        offset = 2**15
    elif dname=='Liberty08' or dname=='DASARA':
        amp_scape = (2.5/65535)*(10**(134/20))
        offset = 0
    if x.ndim==1: # check if vector
        x = x[:,np.newaxis].T
    if x.shape[0]>x.shape[1]: # check if channel axis first
        x = x.T
        
    y = np.zeros((x.shape[0], x.shape[1]))
    for ch in range(x.shape[0]):
        y[ch,:] = amp_scale*sp.lfilter(b_filt, a_filt, (x[ch,:] - np.array((offset,))))
    
    return y
        