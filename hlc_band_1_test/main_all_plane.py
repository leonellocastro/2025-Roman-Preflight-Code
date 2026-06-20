import os

import numpy as np
import matplotlib.pyplot as plt
import proper
import roman_preflight_proper

from roman_hlc_band1_all_planes import hlc

wavelength = 0.575e-6
grid_size = 1024
diam_telescope = 2.363114
beam_ratio = 309.0 / 1024.0
output_dim = 256
final_sampling_lam0 = 0.1
rootname = "hlc_ni_3e-8"
use_polmap = False
polaxis = 10

data_root = os.path.join(
    "C:\\",
    "Users",
    "leone",
    "OneDrive",
    "Documents",
    "GitHub",
    "2025-Roman-Preflight-Code",
    "roman_preflight_proper_public_v2.0.1_python",
    "roman_preflight_proper",
    "preflight_data",
    "hlc_20190210b",
)

pupil = os.path.join(data_root, "pupil.fits")
fpm_real = os.path.join(data_root, "hlc_fpm_trans_0.57500000um_real.fits")
fpm_imag = os.path.join(data_root, "hlc_fpm_trans_0.57500000um_imag.fits")
lyot_stop = os.path.join(data_root, "lyot_rotated.fits")

dm1 = os.path.join(roman_preflight_proper.lib_dir, "examples", rootname + "_dm1_v.fits")
dm2 = os.path.join(roman_preflight_proper.lib_dir, "examples", rootname + "_dm2_v.fits")

if not os.path.exists(fpm_real):
    fpm_real = os.path.join(data_root, "hlc_fpm_trans_0.57600000um_real.fits")
    fpm_imag = os.path.join(data_root, "hlc_fpm_trans_0.57600000um_imag.fits")


unocculted_field, unocculted_sampling, unocculted_wavefront = hlc(
    wavelength,
    diam_telescope,
    scale_occulter=0,
    grid_size=grid_size,
    beam_ratio=beam_ratio,
    f_lens=None,
    pupil=pupil,
    fpm_real=fpm_real,
    fpm_imag=fpm_imag,
    dm1=dm1,
    dm2=dm2,
    lyot_stop=lyot_stop,
    output_dim=output_dim,
    final_sampling_lam0=final_sampling_lam0,
    use_polmap=use_polmap,
    polaxis=polaxis,
)

final_field, final_sampling, final_wavefront = hlc(
    wavelength,
    diam_telescope,
    scale_occulter=1,
    grid_size=grid_size,
    beam_ratio=beam_ratio,
    f_lens=None,
    pupil=pupil,
    fpm_real=fpm_real,
    fpm_imag=fpm_imag,
    dm1=dm1,
    dm2=dm2,
    lyot_stop=lyot_stop,
    output_dim=output_dim,
    final_sampling_lam0=final_sampling_lam0,
    use_polmap=use_polmap,
    polaxis=polaxis,
)








unocculted_psf = np.abs(unocculted_field) ** 2
final_psf = np.abs(final_field) ** 2
max_unocculted = np.max(unocculted_psf)
max_final = np.max(final_psf)
psf_peak_ratio = max_final / max_unocculted

# Calculate the precise coordinate conversion factor
final_sampling = proper.prop_get_sampling_radians(final_wavefront)
sampling_in_lamD = final_sampling / (wavelength / diam_telescope)

# Create a symmetric coordinate system centered on 0
pixel_indices = np.arange(output_dim) - (output_dim / 2.0)
lamD_coordinates = pixel_indices * sampling_in_lamD

# Define the exact extent window for plot boundaries
extent_bounds = [lamD_coordinates[0], lamD_coordinates[-1], lamD_coordinates[0], lamD_coordinates[-1]]

print(f"DM1 file: {dm1}")
print(f"DM2 file: {dm2}")
print(f"FPM real file: {fpm_real}")
print(f"FPM imag file: {fpm_imag}")
print(f"Lyot stop file: {lyot_stop}")
print(f"Polarization enabled: {use_polmap} (polaxis={polaxis})")
print(f"Output sampling (m/pix): {final_sampling:.6e}")
print(f"Peak intensity ratio (occulted / unocculted): {psf_peak_ratio:.2e}")

# Plot the normalized intensity of the star field with occulter, using the no-occultation case as the normalization reference (log scale)
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(final_psf / max_unocculted + 1e-12), origin="lower", extent=extent_bounds, cmap='magma')
plt.colorbar(label='Normalized Contrast $\\log_{10}(I/I_0)$')
plt.xlabel('X Separation [$\\lambda_0/D$]', fontsize=20)
plt.ylabel('Y Separation [$\\lambda_0/D$]', fontsize=20)
plt.title("Star Field with Occulter", fontsize=20)
plt.show()