import numpy as np
import matplotlib.pyplot as plt
import hcipy as hp

# make a 1D contrast curve plot along the horizontal center line (y=0)

def contrast_curve(wavefront_star,focal_grid,prop,wavefront_focal_after_occulter_total_intensity,planet_offset_x,sigma_lambda_d):
    # 1. Calculate the true normalization factor (Peak Intensity of the unocculted star)
    # This uses the 'prop' (FraunhoferPropagator) and 'wavefront_star' defined earlier.
    I_star_unocculted = prop.forward(wavefront_star).intensity
    I_star_peak = I_star_unocculted.max()

    # 2. Calculate Log Contrast relative to the unocculted star peak
    # This provides the standard contrast definition: I_final / I_star_peak
    log_contrast = np.log10(wavefront_focal_after_occulter_total_intensity/I_star_peak)

    # 3. Extract the slice parameters
    # The focal_grid is a 1D hcipy Field, which needs reshaping and slicing logic.
    grid_dimension = int(np.sqrt(log_contrast.size))
    center_index = grid_dimension // 2

    # Reshape the data and coordinates to 2D
    log_contrast_2D = log_contrast.reshape((grid_dimension, grid_dimension))
    x_focal_2D = focal_grid.x.reshape((grid_dimension, grid_dimension))

    # Extract the middle row slice (y=0)
    contrast_slice = log_contrast_2D[center_index, :]

    # Get the corresponding x-coordinates (in lambda/D)
    x_slice = x_focal_2D[center_index, :]

    # 4. Filter for the right half (x/D > 0)
    # We start the slice from the center index (where x/D is 0) to the end.
    x_plot = x_slice[center_index:]
    I_plot = contrast_slice[center_index:]

    # 5. Plot the 1D Contrast Curve

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_plot, I_plot,label=f'Contrast Profile',linewidth=2,color="#54297d")

    # Highlight the planet location
    ax.axvline(x=planet_offset_x, color='r', linestyle='--', alpha=0.6, label=f'Planet at {planet_offset_x:.1f} $\\lambda/D$')

    # Styling (Font size for title, labels, numbers, and legend)

    ax.set_ylim(-20, -12) # Set limits for typical coronagraph contrast
    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_xlabel('Angular Separation ($\\lambda/D$)', fontsize=20)
    ax.set_ylabel('Log Contrast ($log_{10}(I / I_{star,peak})$)', fontsize=20)
    ax.set_title(f"Coronagraph Contrast Curve ($\\sigma = {sigma_lambda_d:.1f}$ $\\lambda/D$)", fontsize=20)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()