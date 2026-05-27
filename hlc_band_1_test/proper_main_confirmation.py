import cgisim
import proper
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import roman_preflight_proper as rp

def run_cgisim_verification():
    # 1. Configuration (Matching testsim_hlc.py)
    cgi_mode = 'excam'  # Executive Camera mode
    cor_type = 'hlc'    # Hybrid Lyot Coronagraph
    bandpass = '1'      # Band 1 (575 nm)
    polaxis = -10       # Mean X+Y polarization

    # 2. Load DM files (Using the standard library examples)
    # These are the same files used in your testsim_hlc_iterations.py
    dm1 = proper.prop_fits_read(rp.lib_dir + '/examples/hlc_ni_3e-8_dm1_v.fits')
    dm2 = proper.prop_fits_read(rp.lib_dir + '/examples/hlc_ni_3e-8_dm2_v.fits')

    # 3. Parameters for Occulted Image (The "Dark Hole")
    params_occulted = {
        'use_errors': 1, 
        'use_dm1': 1, 
        'dm1_v': dm1, 
        'use_dm2': 1, 
        'dm2_v': dm2,
        'use_fpm': 1    # Mask is IN
    }

    print("Computing occulted coronagraphic field...")
    # rcgisim handles the stellar spectrum (A0V) and magnitude automatically
    a0_occulted, a0_counts = cgisim.rcgisim(
        cgi_mode, cor_type, bandpass, polaxis, params_occulted, 
        star_spectrum='a0v', star_vmag=2.0
    )

    # 4. Parameters for Unocculted PSF (For Normalization)
    params_unocculted = params_occulted.copy()
    params_unocculted['use_fpm'] = 0  # Mask is OUT

    print("Computing unocculted PSF for normalization...")
    a0_psf, a0_psf_counts = cgisim.rcgisim(
        cgi_mode, cor_type, bandpass, polaxis, params_unocculted, 
        star_spectrum='a0v', star_vmag=2.0
    )

    # 5. Calculate Normalized Intensity (NI)
    max_psf = np.max(a0_psf)
    ni_image = a0_occulted / max_psf

    # 6. Plotting (Using the 100-pixel trim from your examples)
    plt.figure(figsize=(8, 8))
    # Using the jet colormap and LogNorm as seen in testsim_hlc.py
    im = plt.imshow(
        rp.trim(ni_image, 100), 
        norm=LogNorm(vmin=1e-10, vmax=1e-5), 
        cmap='jet', 
        origin='lower'
    )
    plt.colorbar(im, label='Normalized Intensity')
    plt.title('CGISim HLC Band 1: Normalized Intensity')
    plt.show()

if __name__ == "__main__":
    run_cgisim_verification()

"""
import os

import numpy as np
import proper
import roman_preflight_proper

from proper_confirmation import hlc

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


unocculted_field, unocculted_sampling = hlc(
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

final_field, final_sampling = hlc(
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
psf_peak_ratio = np.max(final_psf) / np.max(unocculted_psf)

print(f"DM1 file: {dm1}")
print(f"DM2 file: {dm2}")
print(f"FPM real file: {fpm_real}")
print(f"FPM imag file: {fpm_imag}")
print(f"Lyot stop file: {lyot_stop}")
print(f"Polarization enabled: {use_polmap} (polaxis={polaxis})")
print(f"Output sampling (m/pix): {final_sampling:.6e}")
print(f"Peak intensity ratio (occulted / unocculted): {psf_peak_ratio:.2e}")
"""