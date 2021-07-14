def phase_intensity(Sx1, Sx2, Sx3): # starting for gsi file

    import numpy as np
    from math import atan2
    
    # check for input dimensionality
    if (Sx1.shape[0]>Sx1.shape[1]) or (Sx2.shape[0]>Sx2.shape[1]) or (Sx3.shape[0]>Sx3.shape[1]):
        print('Please ensure the channel dimension is first, followed by sample dimension.')
        return
    
    Ix = Sx1*Sx2 # x-component
    Iy = Sx1*Sx3 # y-component
    
    active_phase = np.rad2deg(atan2( np.sqrt(np.imag(Ix)**2+np.imag(Iy)**2), \
                                    np.sqrt(np.real(Ix)**2+np.imag(Iy)**2) ))
      
    reactive_phase = np.rad2deg(atan2( np.sqrt(np.real(Ix)**2+np.real(Iy)**2), \
                                      np.sqrt(np.imag(Ix)**2+np.imag(Iy)**2) ))
        
    return active_phase, reactive_phase