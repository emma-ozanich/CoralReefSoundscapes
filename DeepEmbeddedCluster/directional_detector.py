def directional_detector(x,y, a_grid):
    # current designed to check 2 sensors, x & y
    # a_grid is a grid of indices
    
    # parallelize?

    import numpy as np
    
    # run in parallel
    import itertools

    def loop_over_grid(i, j, x, a_grid, y):
        daz = np.round((a_grid[1]-a_grid[0])) # delta azimuth for grid
        bincheck = ( (x >= (a_grid[i]-daz))*(x <= (a_grid[i]+daz)) ) # # check first sector
        bincheck = bincheck*((y >= (a_grid[j]-daz))*(y <= (a_grid[j]+daz))) # # check second sector
        
        return bincheck # return binary matrix for this combo of sectors
        
    grid_ind = np.arange(len(a_grid)) # indices of grid 
    Ngrid = len(grid_ind)
    det = [loop_over_grid(i,j,x,a_grid,y) for i,j in itertools.permutations(grid_ind,2)]
    det2D = sum(det) #sum detections across sector (!new)
            
    return det2D, det
