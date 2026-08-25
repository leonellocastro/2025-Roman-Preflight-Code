import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from fig2img import fig2img
from gaussian_occulter import gaussian_occulter_generator
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input parameters
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 6 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.05
pupil_grid_size = 1024

# Create grids
pupil_grid = hp.make_pupil_grid(pupil_grid_size, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(2, 16, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = hp.make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
ratio = 0.7
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)
Lyot_Coronagraph = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)

# SM parameters
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)

# Define the modes to be controlled on each segment (Piston, Tip, Tilt)
# modes=hp.make_zernike_basis(3, 1, pupil_grid, starting_mode=1)
# 3 modes: Zernike 1 (Piston, though usually 0), Zernike 2 (Tip), Zernike 3 (Tilt)
segment_modes = hp.make_zernike_basis(3, segment_flat_to_flat, pupil_grid, starting_mode=1)

# segments is a list of Field objects (the influence functions)
aper_field, segments = hp.make_hexagonal_segmented_aperture(num_rings=3,
                                                     segment_flat_to_flat=segment_flat_to_flat,
                                                     gap_size=gap_size,
                                                     starting_ring=1,
                                                     return_segments=True,
                                                     ) # <<< TTP UPGRADE
aper = hp.evaluate_supersampled(aper_field, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

hsm = hp.SegmentedDeformableMirror(segments)

# Plot the segmented deformable mirror
plt.title('HSM')
plt.xlabel('x / D')
plt.ylabel('y / D')
hp.imshow_field(hsm.surface, grid_units=pupil_diameter, mask=aper, cmap='RdBu', vmin=-5e-9, vmax=5e-9)
plt.show()

num_modes = len(hsm.influence_functions)

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        hsm.actuators = actuators

    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

    img = prop(Lyot_Coronagraph(hsm(wf)))

    return img

# --- REFERENCE IMAGE (Uncoronagraphed Star) ---
img_ref = prop(hp.Wavefront(aperture, wavelength)).intensity
img_ref_max = img_ref.max() # Use the max for normalization

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

jacobian = get_jacobian_matrix(get_image, dark_zone, num_modes)

rcond_tikhonov = 0.0001
efc_iterations = 500
def run_efc(get_image, dark_zone, num_modes, jacobian, rcond):
    # Calculate EFC matrix
    efc_matrix = hp.inverse_tikhonov(jacobian, rcond)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    # Get initial image and contrast before loop starts
    initial_img = get_image(current_actuators)
    initial_contrast = np.mean(initial_img.intensity[dark_zone] / img_ref_max)
    print(f"Initial Contrast (Iteration 0): {initial_contrast:.2e}")
    print(f"Starting EFC loop for {efc_iterations} iterations...")

    for i in range(efc_iterations):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y

    return actuators, electric_fields, images

actuators, electric_fields, images = run_efc(get_image, dark_zone, num_modes, jacobian, rcond_tikhonov)


# --- ANIMATION SETUP ---
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe' # <<-- Check this path!

fig = plt.figure(figsize=(12, 10))
filename = 'sm_efc_gaussian_animation_final_manual.mp4'
fps_rate = 5
dpi_quality = 150
anim = FFMpegWriter(fps=fps_rate, metadata=dict(artist='HCIPy SM EFC Simulation'))
num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]
electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)
iteration_range = np.linspace(0, num_iterations - 1, num_iterations, dtype=int)

def make_animation_sm_efc(iteration):
    """Drawing function for a single frame of the SM EFC loop."""
    
    plt.clf()

    # Change font size for all plots (title, x axis, y axis, numbers, and colorbar labels)
    for label in plt.get_current_fig_manager().canvas.figure.get_axes():
        label.title.set_size(20)
        label.xaxis.label.set_size(20)
        label.yaxis.label.set_size(20)
        label.tick_params(axis="both", which="major", labelsize=10)

    # 1. Electric field subplot
    plt.subplot(2, 2, 1)
    plt.title('Electric field', fontsize=20)
    plt.xlabel('$\\lambda/D$', fontsize=20)
    plt.ylabel('$\\lambda/D$', fontsize=20)
    electric_field = electric_fields[iteration] / np.sqrt(img_ref) 
    hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 2. Intensity image subplot
    plt.subplot(2, 2, 2)
    plt.title('Intensity image (Log Contrast)', fontsize=20)
    plt.xlabel('$\\lambda/D$', fontsize=20)
    plt.ylabel('$\\lambda/D$', fontsize=20)
    hp.imshow_field(np.log10(images[iteration] / img_ref), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=2)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 3. SM Piston Map subplot
    plt.subplot(2, 2, 3)
    hsm.actuators = actuators[iteration]
    plt.title('SM Piston Map in nm', fontsize=20)
    plt.xlabel('X (m)', fontsize=20)
    plt.ylabel('Y (m)', fontsize=20)
    hp.imshow_field(hsm.surface * 1e9, grid_units=pupil_diameter, mask=aper, cmap='RdBu', vmin=-5, vmax=5)
    plt.colorbar()

    # 4. Average contrast plot
    plt.subplot(2, 2, 4)
    plt.title('Average Contrast', fontsize=20)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Contrast ($log_{10}(I/I_{total})$)", fontsize=20)
    plt.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    plt.xlim(0, num_iterations)
    plt.yscale('log')
    plt.ylim(1e-11, 1e-5)
    plt.grid(color='0.5', linestyle='--')


    plt.suptitle('SM EFC Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')

# --- VIDEO SAVING LOOP ---
print("\n--- Video Saving ---")
try:
    anim.setup(fig, filename, dpi=dpi_quality)

    print(f"Starting animation rendering ({num_iterations} frames)...")
    for i in iteration_range:
        if (i+1) % 5 == 0 or i == 0:
            print(f"Rendering frame {i+1} / {num_iterations}")
        make_animation_sm_efc(i) 
        anim.grab_frame()     
        
except Exception as e:
    print("\n--- CRITICAL ERROR DURING SAVING ---")
    print(f"Error: {e}")
    print(f"If this is a FileNotFoundError, please ensure 'C:\\ffmpeg\\bin\\ffmpeg.exe' is the correct path.")
    
finally:
    anim.finish()
    print(f"\nAnimation process finished. Output file saved to '{filename}'.")
    plt.show()

# Make plots of final results

# Final Intensity Image
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title('Final Intensity Image (Log Contrast)')
hp.imshow_field(np.log10(images[-1] / img_ref), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=2)
plt.colorbar()
hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Average Contrast Plot
plt.subplot(1,2,2)
plt.title('Average Contrast in Dark Zone')
plt.xlabel("EFC Iteration")
plt.ylabel("Average Contrast")
plt.plot(range(len(average_contrast)), average_contrast, 'o-')
plt.yscale('log')
plt.ylim(1e-11, 1e-5)
plt.grid(color='0.5', linestyle='--')
plt.suptitle('Final Results after %d Iterations' % num_iterations, fontsize='x-large')
fig_intensity = plt.gcf()
img_intensity = fig2img(fig_intensity)
img_intensity.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/Images/sm_dm_final_intensity.png')
plt.show()

# Final SM Piston Map
plt.figure(figsize=(12,6))
hsm.actuators = actuators[-1]
plt.title('Final SM Piston Map in nm')
hp.imshow_field(hsm.surface * 1e9, grid_units=pupil_diameter, mask=aper, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar()

# Save the DM surfaces
fig_dm = plt.gcf()
img_dm = fig2img(fig_dm)
img_dm.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/Images/sm_dm_final_dm_surfaces.png')
plt.show()

# Print the initial and final average contrasts
initial_contrast = average_contrast[0]
final_contrast = average_contrast[-1]
print(f"Initial Average Contrast: {initial_contrast:.2e}")
print(f"Final Average Contrast: {final_contrast:.2e}")