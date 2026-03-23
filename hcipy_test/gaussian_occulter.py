import numpy as np
import hcipy as hp
def gaussian_occulter_generator(grid,sigma):
    x = grid.x
    y = grid.y
    r = np.sqrt(x**2 + y**2)
    sigma_val = (r/sigma)
    transmission = 1.0 - np.exp(-0.5 * sigma_val**2)
    transmission_field = hp.Field(transmission,grid)
    return transmission_field