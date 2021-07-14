def vector_direction(Sx1, Sx2, Sx3, ref_bearing, F, Fadj): # starting for gsi file
    # Sx1, Sx2, Sx3: spectrograms (stft) for pressure, x- and y- channels of DASAR
    # ref_bearing: reference bearing for instrument
    # F: F-vector corresponding to stft
    # Fadj: frequency at which to adjust reactive/active intensity due to phase shift

    import numpy as np
    
    def bnorm(direction): # # Fixes azimuth wrap-around after adjustment
        I = (direction >= 360)
        direction[I] = direction[I] - 360 # account for positive wrap-around
        J = (direction <0)
        direction[J] = direction[J] + 360 # account for negative wrap-around
        
        return direction
    
    def math_to_geometric(direction):
        direction = 90 - direction
        I = (direction < 0 ) 
        direction[I] = direction[I] + 360
        
        return direction
    
    # check for input dimensionality
   # if (Sx1.shape[0]>Sx1.shape[1]) or (Sx2.shape[0]>Sx2.shape[1]) or (Sx3.shape[0]>Sx3.shape[1]):
   #     print('Please ensure the channel dimension is first, followed by sample dimension.')
   #     return
    
    Ix = Sx1*np.conjugate(Sx2) # x-component
    Iy = Sx1*np.conjugate(Sx3) # y-component
    
    active_directionality = bnorm( math_to_geometric( np.rad2deg(np.arctan2(np.real(Iy), np.real(Ix))) ) + ref_bearing )
    reactive_directionality =  bnorm( math_to_geometric( np.rad2deg( np.arctan2(np.imag(Iy), np.imag(Ix))) ) + ref_bearing ) 
    
    if len(Fadj): # if we need to adjust for phase drift per freq
        # this is a rough adjustment Aaron suggested for his instruments
        I = np.argmin(np.abs(F - Fadj))
        active_directionality[(I+1):,:] = reactive_directionality[(I+1):,:]
        
    
    return active_directionality, reactive_directionality