def intensity_energy(Sx1, Sx2, Sx3, F, Fadj): # starting for gsi file
    
    import numpy as np
    
    # check for input dimensionality
    if (Sx1.shape[0]>Sx1.shape[1]) or (Sx2.shape[0]>Sx2.shape[1]) or (Sx3.shape[0]>Sx3.shape[1]):
        print('Please ensure the channel dimension is first, followed by sample dimension.')
        return
    
    Ix = Sx1*Sx2 # x-component
    Iy = Sx1*Sx3 # y-component
    
    active_intensity = np.sqrt( np.real(Ix)**2 + np.real(Iy)**2 )
    reactive_intensity = np.sqrt( np.imag(Ix)**2 + np.imag(Iy)**2 )
    
    if len(Fadj): # if we need to adjust for phase drift per freq
        # this is a rough adjustment Aaron suggested for his instruments
        I = np.argmin(np.abs(F - Fadj))
        active_intensity[(I+1):,:] = reactive_intensity[(I+1):,:]
    
    energy_density = 0.5*( (np.abs(Sx1)**2 ) + np.abs(Sx2)**2 + np.abs(Sx3)**2 ) # pressure autospectrum + velocity autospectrum
    
    active_ItoE = active_intensity/energy_density
    reactive_ItoE = reactive_intensity/energy_density
    
    return active_ItoE, reactive_ItoE, active_intensity, reactive_intensity

