import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import proper
from scipy.interpolate import interp1d
from roman_preflight_proper import ffts, mft2, polmap, trim
import roman_preflight_proper as rp
from proper_simulation_version_3 import propagate_hlc_system

# --- HLC BAND 1 PHYSICAL CONSTANTS ---
HLC_BAND1_LAMBDA0_M = 0.575e-6
HLC_BAND1_PUPIL_DIAM_PIX = 309.0
HLC_BAND1_GRID_SIZE = 1024
HLC_BAND1_BEAM_RATIO = HLC_BAND1_PUPIL_DIAM_PIX / HLC_BAND1_GRID_SIZE
HLC_BAND1_DM_SAMPLING_M = 0.9906e-3
HLC_BAND1_DM1_XC_ACT = 23.5
HLC_BAND1_DM2_XC_ACT = 23.5 - 0.1
HLC_BAND1_DM_YC_ACT = 23.5
HLC_BAND1_DM_XTILT_DEG = 9.65
HLC_BAND1_DM_YTILT_DEG = 0.0
HLC_BAND1_DM_ZTILT_DEG = 0.0
HLC_BAND1_M_PER_LAMD_575NM_AT_FPM = 1.8541536e-05
HLC_BAND1_M_PER_LAMD_575NM_AT_FIELD_STOP = 2.3957964e-05
HLC_BAND1_FIELD_STOP_RADIUS_LAM0 = 9.7

DATA_DIR = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\"
    
pupil_path = DATA_DIR + "pupil.fits"
fpm_real_path = DATA_DIR + "hlc_fpm_trans_0.54625000um_real.fits"
fpm_imag_path = DATA_DIR + "hlc_fpm_trans_0.54625000um_imag.fits"
lyot_path = DATA_DIR + "lyot.fits"
dm1_path = DATA_DIR + "hlc_dm1.fits"
dm2_path = DATA_DIR + "hlc_dm2.fits"

planet_x_offset = 5
planet_contrast = 0*1e0

# No Occulter

print("--> Simulating Coherent Stellar Wavefront...")
star_field_no_occulter, sampl = propagate_hlc_system(
    HLC_BAND1_LAMBDA0_M, 2.363114, HLC_BAND1_GRID_SIZE, HLC_BAND1_BEAM_RATIO,
    pupil_path, fpm_real_path, fpm_imag_path, dm1_path, dm2_path, lyot_path,
    is_planet=False, occulter_applied=False
)

# Take detector plane image maxium as normalization reference

norm = np.max(np.abs(star_field_no_occulter)**2)

# Run Without Occulter

star_field_with_occulter, sampl = propagate_hlc_system(
    HLC_BAND1_LAMBDA0_M, 2.363114, HLC_BAND1_GRID_SIZE, HLC_BAND1_BEAM_RATIO,
    pupil_path, fpm_real_path, fpm_imag_path, dm1_path, dm2_path, lyot_path,
    is_planet=False, occulter_applied=True
)

# Plot the normalized intensity of the star field with occulter, using the no-occultation case as the normalization reference

plt.figure(figsize=(10, 8))
plt.imshow(np.log10(np.abs(star_field_with_occulter)**2 / norm + 1e-12), origin="lower", cmap='magma')
plt.colorbar(label="Log10 Normalized Intensity")
plt.title("HLC Band 1 Simulation: Stellar Field with Occulter (Normalized to No-Occulter Peak)", fontsize=14)
plt.xlabel("Angular Separation [$\\lambda_0/D$]", fontsize=12)
plt.ylabel("Angular Separation [$\\lambda_0/D$]", fontsize=12)
plt.show()