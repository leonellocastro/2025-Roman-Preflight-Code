import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import warnings
from gaussian_occulter import gaussian_occulter_generator
# from animation import animate_coronagraph
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- ANIMATION PARAMETERS ---
# Varying planet separation from 2 to 25 lambda/D
ratio_range = np.linspace(0, 1, 21)
vmin = -16 # Log contrast minimum for plotting (adjust based on your sqrt_contrast)
vmax = -8  # Log contrast maximum for plotting

focal_grid = hp.make_focal_grid(q=8, num_airy=16)

# Initialize the figure and image plot
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)
# Create a blank plot initially for the animation handle
im_handle = hp.imshow_field(
    np.zeros(focal_grid.size), 
    grid=focal_grid, 
    cmap='inferno', 
    vmin=vmin, 
    vmax=vmax,
    ax=ax
)
plt.colorbar(im_handle, label='Contrast ($log_{10}(I/I_{total})$)')
title = ax.set_title("")
ax.set_xlabel('x / D')
ax.set_ylabel('y / D')

fps = 2 # Frames per second for the output video
filename = 'lyot_stop_animation.mp4' # Output filename

def animate_coronagraph_lyot_stop_ratio(ratio):
    aperture_scale = 1.5
    grid_size = 256
    local_grid_size = 256 # Defined here for use in reshape/normalization
    pupil_grid = hp.make_pupil_grid(grid_size,aperture_scale)
    diameter = 1 # meters

    telescope_pupil_generator = hp.make_circular_aperture(diameter)

    telescope_pupil = telescope_pupil_generator(pupil_grid)

    # define propagator (pupil to focal)
    focal_grid = hp.make_focal_grid(q=8, num_airy=16)
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

    # obtain wavefront at telescope pupil plane for the star
    wavefront_star = hp.Wavefront(telescope_pupil)

    # obtain wavefront at telescope pupil plane for the planet
    contrast = 1e-10
    sqrt_contrast = np.sqrt(contrast) # Planet-to-star contrast (note: sqrt because we are working with the electric field)

    # Planet offset in units of lambda/D
    planet_offset_x = 15
    planet_offset_y = 0
    planet_offset_x = planet_offset_x/diameter
    planet_offset_y = planet_offset_y/diameter
    wavefront_planet = hp.Wavefront(sqrt_contrast * telescope_pupil * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))

    # obtain total wavefront intensity at pupil plane
    wavefront_total = hp.Wavefront(wavefront_star.electric_field + wavefront_planet.electric_field)

    # obtain the total wavefront at focal plane
    focal_total = prop.forward(wavefront_total)

    # obtain maximum focal plane intensity
    focal_total_max = np.max(focal_total.intensity)

    # create the Gaussian occulter mask
    sigma_lambda_d = 5
    occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
    occulter_mask = hp.Field(occulter_mask,focal_grid)

    # create the occulter mask and Lyot Stop in the Lyot Coronagraph
    # ratio = 0.8 # Lyot Stop diameter ratio
    lyot_stop_generator = hp.make_circular_aperture(ratio*diameter) # percentage of the telescope diameter
    lyot_stop_mask = lyot_stop_generator(pupil_grid)
    prop_lyot = hp.LyotCoronagraph(pupil_grid,occulter_mask,lyot_stop_mask)
    occulter_lyot_wavefront_pupil = prop_lyot.forward(wavefront_total)

    # propagate the wavefront to the focal plane
    wavefront_focal_after_occulter_total = prop.forward(occulter_lyot_wavefront_pupil)
    wavefront_focal_after_occulter_total_intensity = wavefront_focal_after_occulter_total.intensity

    # Normalize and convert to log scale
    I_norm = wavefront_focal_after_occulter_total_intensity / focal_total_max
    log_I_norm = np.log10(I_norm)
    
    # CRITICAL FIX 4: Reshape is required for matplotlib's set_data()
    # Use local_grid_size, which is 256
    reshaped_data = log_I_norm.reshape((local_grid_size, local_grid_size))
    
    # Update the plot handle with the new data
    im_handle.set_data(reshaped_data)
    
    # Update the title
    title.set_text(r"Coronagraphic Image (Lyot Stop Ratio: " + f"{ratio:.2f}" + r")")

    # Return the updated artists for blitting
    return im_handle,

# Create the animation over planet_offset_x variable
ani = FuncAnimation(
    fig,
    animate_coronagraph_lyot_stop_ratio, 
    frames=ratio_range, # Use the list of separations as frames
    blit=False, 
    interval=1000 # milliseconds between frames
)

# --- VIDEO SAVING LOGIC ---
# 'C:/path/to/ffmpeg.exe' placeholder below
# with the ACTUAL path to FFmpeg executable file.
ffmpeg_path = 'C:/ffmpeg/bin/ffmpeg.exe'

plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

try:
    # Set up the writer, passing the explicit path
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='HcIPy Simulation'))

    print(f"Starting to save animation to {filename}...")
    # Save the animation. This will take some time.
    ani.save(filename, writer=writer)
    print(f"Animation successfully saved to {filename}")

except FileNotFoundError:
    print("\n--- ERROR ---")
    print(f"Failed to find FFmpeg at the specified path: {ffmpeg_path}")
    print("Please install FFmpeg and update the 'ffmpeg_path' variable in the script with the correct location.")
except Exception as e:
    print(f"An error occurred during saving: {e}")
plt.show()