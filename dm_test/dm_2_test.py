import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from fig2img import fig2img
from gaussian_occulter import gaussian_occulter_generator
import os
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input parameters
pupil_diameter = 1 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across * pupil_diameter
aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9
DM_STROKE_LIMIT = 5e-7 # meters

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 6 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.05

# Create grids
grid_size = 256
q = 8
num_airy = 16
pupil_grid = hp.make_pupil_grid(grid_size, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(q, num_airy, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# Create aperture and dark zone
aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

# Create dark zone
dark_zone = hp.make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# create the occulter mask and Lyot Stop in the Lyot Coronagraph
ratio = 0.7 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)
coronagraph = hp.LyotCoronagraph(pupil_grid,occulter_mask,lyot_stop_mask)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
# aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, exponent=-3)

# This uses the 32x32 continuous DM model, matching your goal
influence_functions = hp.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror_1 = hp.DeformableMirror(influence_functions)
deformable_mirror_2 = hp.DeformableMirror(influence_functions)

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror_1.actuators = actuators[:len(influence_functions)]
        deformable_mirror_2.actuators = actuators[len(influence_functions):]
    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

    img = prop(deformable_mirror_2(coronagraph(deformable_mirror_1(wf))))

    return img

img_ref = prop(hp.Wavefront(aperture, wavelength)).intensity

def get_jacobian_matrix(get_image, dark_zone, num_modes):
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2)

    for i, mode in enumerate(np.eye(num_modes)):
        response = 0

        for amp in amps:
            response += amp * get_image(mode * amp, include_aberration=False).electric_field

        response /= np.var(amps)
        response = response[dark_zone]

        responses.append(np.concatenate((response.real, response.imag)))

    jacobian = np.array(responses).T
    return jacobian

jacobian = get_jacobian_matrix(get_image, dark_zone, 2 * len(influence_functions))
rcond = 0.025

def run_efc(get_image, dark_zone, num_modes, jacobian, rcond):
    # Calculate EFC matrix
    efc_matrix = hp.inverse_tikhonov(jacobian, rcond)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    # Keeping iterations low for stability
    NUM_ITERATIONS = 1000
    
    for i in range(NUM_ITERATIONS):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y
        
        # ENFORCE LIMIT: Uses the DM_STROKE_LIMIT variable
        current_actuators = np.clip(current_actuators, -DM_STROKE_LIMIT, DM_STROKE_LIMIT)

    return actuators, electric_fields, images, NUM_ITERATIONS

# Get the results and the actual number of iterations used
actuators, electric_fields, images, num_iterations = run_efc(get_image, dark_zone, 2 * len(influence_functions), jacobian, rcond)





# --- Animation Rendering Block ---

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# Use a figure size and dpi that is friendly to memory 
fig = plt.figure(figsize=(9, 7)) 

anim = FFMpegWriter(fps=5, metadata=dict(artist='HCIPy EFC Loop'))

average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.0), True)

iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)

def make_animation_1dm(iteration):
    deformable_mirror_1.actuators = actuators[iteration][:len(influence_functions)]
    deformable_mirror_2.actuators = actuators[iteration][len(influence_functions):]
    # Clear the entire figure before drawing new subplots
    fig.clf()
    
    # Re-establish subplot layout
    
    # 1. Electric Field
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.set_title('Focal Plane Electric Field')
    # ax1.set_xlabel('x/D')
    # ax1.set_ylabel('y/D')
    # electric_field = electric_fields[iteration] / np.sqrt(img_ref.max())
    # hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution, ax=ax1)
    # hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white', ax=ax1)

    # 1. Intensity Image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Intensity Image')
    ax1.set_xlabel('$\\lambda/D$')
    ax1.set_ylabel('$\\lambda/D$')
    log_intensity = np.log10(images[iteration] / img_ref.max())
    img = hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5, ax=ax1)
    plt.colorbar(img, ax=ax1)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white', ax=ax1)

    # 2. Average Contrast
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Average Contrast')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Contrast ($I/I_{max}$)')
    ax2.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    ax2.set_xlim(0, num_iterations)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-11, 1e-5)
    ax2.grid(color='0.5')

    # 3. DM1 Surface
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('DM1 Surface')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    dm_img = hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax3)
    plt.colorbar(dm_img, ax=ax3, label='DM1 Surface (nm)')

    # 4. DM2 Surface
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('DM2 Surface')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    dm_img = hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax4)
    plt.colorbar(dm_img, ax=ax4, label='DM2 Surface (nm)')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.title.set_size(20)
        ax.xaxis.label.set_size(20)  # Fixed
        ax.yaxis.label.set_size(20)  # Fixed
        ax.tick_params(axis="both", which="major", labelsize=10)

    # Supertitle
    fig.suptitle('Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')
    # Adjust layout to prevent overlap
    fig.tight_layout()

filename = 'efc_loop_animation_dm_1_dm_2.mp4' # Output filename
print("Setting up animation writer...")
anim.setup(fig, filename, dpi=50)

print("Starting animation rendering...")

# 2. Manual Loop for grabbing frames
for i in iteration_range:
    if (i+1) % 5 == 0 or i == 0:
        print(f"Rendering frame {i+1} / {num_iterations}")
    make_animation_1dm(i) 
    anim.grab_frame() # Grab frame inside the loop after drawing

# Finalize the video file
anim.finish()
print(f"Animation '{filename}' saved successfully after {num_iterations} frames.")
plt.close(fig)

# Intensity image for the final iteration

final_iteration = num_iterations - 1

# Plot the last iteration

# 1. Electric Field
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.title('Electric field for last iteration')
# electric_field = electric_fields[final_iteration] / np.sqrt(img_ref.max())
# hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
# hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Intensity Image
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title('Intensity image for last iteration')
plt.xlabel('x/D')
plt.ylabel('y/D')
log_intensity = np.log10(images[final_iteration] / img_ref.max())
hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
plt.colorbar(label='Contrast ($log_{10}(I/I_{total})$)')
hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Average Contrast
plt.subplot(1, 2, 2)
plt.title('Average Dark Zone Contrast')
plt.xlabel('Iteration')
plt.ylabel('Average Contrast ($log_{10}(I/I_{total})$)')
plt.plot(range(num_iterations), average_contrast, 'o-')
plt.xlim(0, num_iterations)
plt.yscale('log')
plt.ylim(1e-11, 1e-5)
plt.grid(color='0.5')
plt.suptitle('Final Results after %d Iterations' % num_iterations, fontsize='x-large')

fig_intensity = plt.gcf()
img_intensity = fig2img(fig_intensity)
img_intensity.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/Images/dm_1_dm_2_final_intensity.png')
plt.show()

# DM 1 Surface
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('DM1 surface for last iteration')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_1.actuators = actuators[final_iteration][:len(influence_functions)]
hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(label='DM1 Surface (nm)')

# DM 2 Surface
plt.subplot(1, 2, 2)
plt.title('DM2 surface for last iteration')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_2.actuators = actuators[final_iteration][len(influence_functions):]
hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(label='DM2 Surface (nm)')

# Save the DM surfaces
fig_dm = plt.gcf()
img_dm = fig2img(fig_dm)
img_dm.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/Images/final_dm_surfaces_dm_1_dm_2.png')
plt.show()

# Print the initial and final contrast values
print("Contrast for first iteration:", average_contrast[0])
print("Contrast for last iteration:", average_contrast[-1])