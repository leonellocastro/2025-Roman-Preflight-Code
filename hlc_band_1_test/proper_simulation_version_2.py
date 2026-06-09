import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import proper
from scipy.interpolate import interp1d
from roman_preflight_proper import ffts, mft2, polmap, trim
import roman_preflight_proper as rp

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

map_dir = rp.map_dir

def _apply_detector_orientation(wavefront):
    wavefront = np.fliplr(wavefront).copy()
    wavefront = np.rot90(wavefront, 3).copy()
    wavefront = np.roll(wavefront, (1, 1), axis=(0, 1))
    return wavefront

def _glass_index(glass, wavelength_m):
    data_dir = rp.data_dir
    data = np.loadtxt(os.path.join(data_dir, "glass", f"{glass}_index.txt"))
    interp = interp1d(data[:, 0], data[:, 1], kind="cubic")
    return interp(wavelength_m * 1e6)

def _to_from_doublet(wavefront, wavelength_m, dz_to_lens, dz_from_lens, r1_a, r2_a, thickness_a, glass_a, separation_m, r1_b, r2_b, thickness_b, glass_b, surface_name, next_surface_name, error_map_files=' '):
    nglass_a = _glass_index(glass_a, wavelength_m)
    nglass_b = _glass_index(glass_b, wavelength_m)

    f_a = 1.0 / ((nglass_a - 1.0) * (1.0 / r1_a - 1.0 / r2_a + (nglass_a - 1.0) * thickness_a / (nglass_a * r1_a * r2_a)))
    h1_a = -f_a * (nglass_a - 1.0) * thickness_a / (nglass_a * r2_a)
    h2_a = -f_a * (nglass_a - 1.0) * thickness_a / (nglass_a * r1_a)

    f_b = 1.0 / ((nglass_b - 1.0) * (1.0 / r1_b - 1.0 / r2_b + (nglass_b - 1.0) * thickness_b / (nglass_b * r1_b * r2_b)))
    h1_b = -f_b * (nglass_b - 1.0) * thickness_b / (nglass_b * r2_b)
    h2_b = -f_b * (nglass_b - 1.0) * thickness_b / (nglass_b * r1_b)

    proper.prop_propagate(wavefront, dz_to_lens + h1_a, surface_name)
    proper.prop_lens(wavefront, f_a, surface_name + " lens #1")
    if error_map_files != ' ':
        proper.prop_errormap(wavefront, error_map_files[0], WAVEFRONT=True)
    proper.prop_propagate(wavefront, -h2_a + separation_m + h1_b)
    proper.prop_lens(wavefront, f_b, surface_name + " lens #2")
    if error_map_files != ' ' and error_map_files[1] != ' ':
        proper.prop_errormap(wavefront, error_map_files[1], WAVEFRONT=True)
    proper.prop_propagate(wavefront, -h2_b + dz_from_lens, next_surface_name, TO_PLANE=1)

def _apply_hlc_fpm(wavefront, wavelength_m, fpm_real_path, fpm_imag_path):
    """Applies complex (amplitude + phase) FPM with correct coordinate centering."""
    real_map, header = fits.getdata(fpm_real_path, header=True)
    imag_map = fits.getdata(fpm_imag_path)
    
    # CRITICAL FIX: Shift the coordinate centers so the FPM zero-point 
    # aligns perfectly with the centered focused stellar PSF core.
    real_map = proper.prop_shift_center(real_map)
    imag_map = proper.prop_shift_center(imag_map)
    
    fpm_array = real_map + 1j * imag_map
    fpm_mask = (real_map != real_map[0, 0]).astype(int)

    fpm_lam0_m = header["FPMLAM0M"]
    fpm_sampling_lam0divd = header["FPMDX"]
    n = proper.prop_get_gridsize(wavefront)
    m_per_lamdivd = HLC_BAND1_M_PER_LAMD_575NM_AT_FPM * wavelength_m / 575e-9

    wavefront0 = proper.prop_get_wavefront(wavefront)
    wavefront0 = ffts(wavefront0, 1)
    wavefront0 *= fpm_array[0, 0]

    nfpm = fpm_array.shape[0]
    fpm_sampling_lamdivd = fpm_sampling_lam0divd * fpm_lam0_m / wavelength_m
    fpm_sampling_m = fpm_sampling_lamdivd * m_per_lamdivd
    sampling = fpm_sampling_m * (HLC_BAND1_PUPIL_DIAM_PIX / n) / proper.prop_get_sampling(wavefront)

    wavefront_fpm = mft2(wavefront0, sampling, HLC_BAND1_PUPIL_DIAM_PIX, nfpm, -1)
    wavefront_fpm *= fpm_mask * (fpm_array - 1.0)
    wavefront_fpm = mft2(wavefront_fpm, sampling, HLC_BAND1_PUPIL_DIAM_PIX, n, +1)

    wavefront0 += wavefront_fpm
    wavefront0 = ffts(wavefront0, -1)
    wavefront.wfarr[:, :] = proper.prop_shift_center(wavefront0)

def propagate_hlc_system(wavelength, diam, grid_size, beam_ratio, pupil, fpm_real, fpm_imag, dm1, dm2, lyot_stop, 
                         offset_x_lamD=0.0, offset_y_lamD=0.0, is_planet=False, planet_contrast=1.0):
    """Propagates a single coherent wavefront through the entire flight-path optical train."""
    wavefront = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)
    nw = proper.prop_get_gridsize(wavefront)
    sampling = proper.prop_get_sampling(wavefront)
    
    # 1. Primary Entrance Pupil Initialization
    pupil_map = trim(fits.getdata(pupil), grid_size)
    proper.prop_multiply(wavefront, pupil_map)
    
    # Apply Off-Axis Planetary Phase Tilt if evaluating companion beam
    if is_planet:
        x = (np.arange(nw) - nw // 2) * sampling
        y = (np.arange(nw) - nw // 2) * sampling
        xx, yy = np.meshgrid(x, y)
        phase_ramp = np.exp(2j * np.pi * (xx * offset_x_lamD / diam + yy * offset_y_lamD / diam))
        proper.prop_multiply(wavefront, phase_ramp)
        proper.prop_multiply(wavefront, np.sqrt(planet_contrast))
        
    proper.prop_define_entrance(wavefront)

    # 2. Optical Telescope Assembly (OTA) Structure Sequence
    fl_pri = 2.838279206904720
    d_pri_sec = 2.285150508110035 
    fl_sec = -0.654200796568004
    d_sec_pomafold = 2.993753469304728 
    d_pomafold_m3 = 1.680935841598811
    fl_m3 = 0.430216463069001
    d_m3_m4 = 0.943514749358944
    fl_m4 = 0.116239114833590
    d_m4_m5 = 0.429145636743193
    fl_m5 = 0.198821518772608
    d_m5_pupil = 0.716529242927776
    d_m5_ttfold = 0.351125431220770
    
    d_ttfold_fsm = d_m5_pupil - d_m5_ttfold + 0.033609
    d_fsm_oap1 = 0.354826767220001
    fl_oap1 = 0.503331895563883
    d_oap1_fcm = 0.768029932093727
    d_fcm_oap2 = 0.314507535543064
    fl_oap2 = 0.579205571254990
    d_oap2_dm1 = 0.775857408587825
    d_dm1_dm2 = 1.0
    d_dm2_oap3 = 0.394833855161549
    fl_oap3 = 1.217276467668519
    d_oap3_fold3 = 0.505329955078121
    d_fold3_oap4 = 1.158897671642761
    fl_oap4 = 0.446951159052363
    d_oap4_pupilmask = 0.423013568764728
    d_pupilmask_oap5 = 0.408810704327559
    fl_oap5 = 0.548189354706822
    
    fpm_thickness = 0.006363747896388863
    fpm_index = _glass_index("SILICA", wavelength)
    d_fpm_oap6 = fpm_thickness / fpm_index + 0.543766629917668
    fl_oap6 = d_fpm_oap6
    d_oap6_lyotstop = 0.687476361491529
    d_lyotstop_oap7 = 0.401748561745987
    fl_oap7 = 0.708251420923810
    d_oap7_fieldstop = fl_oap7
    d_fieldstop_oap8 = 0.210985170345932 * 0.997651
    fl_oap8 = d_fieldstop_oap8
    d_oap8_pupil = 0.237561587674008
    d_pupil_filter = 0.130
    d_oap8_filter = d_oap8_pupil + d_pupil_filter
    filter_thickness = 0.004016105782012525
    filter_index = _glass_index("SILICA", wavelength)
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095
    d_lens_fold4 = 0.202226
    d_fold4_image = 0.050206330646919

    # Propagate through front OTA optics adding surface errors
    proper.prop_lens(wavefront, fl_pri)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_pri_sec, 'secondary')
    proper.prop_lens(wavefront, fl_sec)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_sec_pomafold, 'POMA FOLD')
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_POMAFOLD_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_pomafold_m3, 'M3')
    proper.prop_lens(wavefront, fl_m3)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M3_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_m3_m4, 'M4')
    proper.prop_lens(wavefront, fl_m4)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M4_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_m4_m5, 'M5')
    proper.prop_lens(wavefront, fl_m5)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M5_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_m5_ttfold, 'TT FOLD')
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_ttfold_fsm, 'FSM')
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_FSM_FLIGHT_coated_measured_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_fsm_oap1, "OAP1")
    proper.prop_lens(wavefront, fl_oap1)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP1_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap1_fcm, "FCM")
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_fcm_oap2, "OAP2")
    proper.prop_lens(wavefront, fl_oap2)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP2_phase_error_V3.0.fits', WAVEFRONT=True)

    # 3. Deformable Mirrors (DMs) Application
    dm_struct = rp.load_cgi_dm_files(dm_files_dir=rp.dm_files_dir, version=rp.dm_version, temp_c=26.0)
    dm1_map = proper.prop_fits_read(dm1)
    dm2_map = proper.prop_fits_read(dm2)

    proper.prop_propagate(wavefront, d_oap2_dm1, "DM1")
    rp.cgi_dm(wavefront, dm_struct, 1, dm1_map, dm_sampling_m=HLC_BAND1_DM_SAMPLING_M, dm_v_quant=110.0/2.**16, 
              dm_xc_act=HLC_BAND1_DM1_XC_ACT, dm_yc_act=HLC_BAND1_DM_YC_ACT, dm_xtilt_deg=HLC_BAND1_DM_XTILT_DEG, dm_ytilt_deg=HLC_BAND1_DM_YTILT_DEG, dm_ztilt_deg=HLC_BAND1_DM_ZTILT_DEG)
    
    proper.prop_propagate(wavefront, d_dm1_dm2, "DM2")
    rp.cgi_dm(wavefront, dm_struct, 2, dm2_map, dm_sampling_m=HLC_BAND1_DM_SAMPLING_M, dm_v_quant=110.0/2.**16, 
              dm_xc_act=HLC_BAND1_DM2_XC_ACT, dm_yc_act=HLC_BAND1_DM_YC_ACT, dm_xtilt_deg=HLC_BAND1_DM_XTILT_DEG, dm_ytilt_deg=HLC_BAND1_DM_YTILT_DEG, dm_ztilt_deg=HLC_BAND1_DM_ZTILT_DEG)

    # 4. Propagate to Focal Plane Mask
    proper.prop_propagate(wavefront, d_dm2_oap3, "OAP3")
    proper.prop_lens(wavefront, fl_oap3)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP3_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap3_fold3, "FOLD_3")
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_fold3_oap4, "OAP4")
    proper.prop_lens(wavefront, fl_oap4)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP4_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap4_pupilmask, "PUPIL_MASK")
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_PUPILFOLD_phase_error_V1.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_pupilmask_oap5, "OAP5")
    proper.prop_lens(wavefront, fl_oap5)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP5_phase_error_V3.0.fits', WAVEFRONT=True)
    
    proper.prop_propagate(wavefront, fl_oap5, "FPM", TO_PLANE=True)

    # Apply true complex FPM matrix (Real + Imaginary) via Matrix Fourier Transform (MFT)
    _apply_hlc_fpm(wavefront, wavelength, fpm_real, fpm_imag)

    # 5. Lyot Stop & Intermediate Field Stop Plane
    proper.prop_propagate(wavefront, d_fpm_oap6, "OAP6")
    proper.prop_lens(wavefront, fl_oap6)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP6_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap6_lyotstop, "LYOT STOP")
    
    lyot_map = trim(fits.getdata(lyot_stop), grid_size)
    proper.prop_multiply(wavefront, lyot_map)

    proper.prop_propagate(wavefront, d_lyotstop_oap7, "OAP7")
    proper.prop_lens(wavefront, fl_oap7)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP7_phase_error_V4.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap7_fieldstop, "FIELD_STOP", TO_PLANE=1)
    
    stop_radius_m = HLC_BAND1_FIELD_STOP_RADIUS_LAM0 * (HLC_BAND1_M_PER_LAMD_575NM_AT_FIELD_STOP * HLC_BAND1_LAMBDA0_M / 575e-9)
    proper.prop_circular_aperture(wavefront, stop_radius_m)

    # 6. Final Camera Focusing Track
    proper.prop_propagate(wavefront, d_fieldstop_oap8, "OAP8")
    proper.prop_lens(wavefront, fl_oap8)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP8_phase_error_V3.0.fits', WAVEFRONT=True)
    proper.prop_propagate(wavefront, d_oap8_filter, "FILTER")
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_FILTER_phase_error_V1.0.fits', WAVEFRONT=True)

    imaging_lens_error_files = [map_dir + 'roman_phasec_LENS_phase_error_V1.0.fits', ' ']
    _to_from_doublet(wavefront, wavelength, d_filter_lens, d_lens_fold4 + d_fold4_image,
                     0.10792660718579995, -0.10792660718579995, 0.003, "S-BSL7R", 0.0005, 1e10,
                     0.10608379812011390, 0.0025, "PBM2R", "IMAGING LENS", "IMAGE", error_map_files=imaging_lens_error_files)

    rsqr = proper.prop_radius(wavefront) ** 2
    field, sampling = proper.prop_end(wavefront, NOABS=True)

    # Color correction phase adjustment matching flight benchmarks
    lam_m = np.array([500, 525, 550, 575, 600, 625, 650, 700, 730, 775, 825, 880]) * 1e-9
    cc = np.array([1.1797, 1.1840, 1.1865, 1.1881, 1.1890, 1.1891, 1.1887, 1.1868, 1.1852, 1.1820, 1.1778, 1.1726])
    interp = interp1d(lam_m, cc, kind="linear", fill_value="extrapolate")
    c = interp(wavelength)
    field *= np.exp((1j * np.pi / wavelength * c) * rsqr) * np.exp(-1j * 0.1)

    field = _apply_detector_orientation(field)
    
    output_dim = 256
    final_sampling_lam0 = 0.1
    mag = (HLC_BAND1_PUPIL_DIAM_PIX / grid_size) / final_sampling_lam0 * (wavelength / HLC_BAND1_LAMBDA0_M)
    sampling = sampling / mag
    field = proper.prop_magnify(field, mag, output_dim, AMP_CONSERVE=True)

    return field, sampling

if __name__ == "__main__":
    # Define local configuration parameters
    DATA_DIR = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\"
    
    pupil_path = DATA_DIR + "pupil.fits"
    fpm_real_path = DATA_DIR + "hlc_fpm_trans_0.54625000um_real.fits"
    fpm_imag_path = DATA_DIR + "hlc_fpm_trans_0.54625000um_imag.fits"
    lyot_path = DATA_DIR + "lyot.fits"
    dm1_path = DATA_DIR + "hlc_dm1.fits"
    dm2_path = DATA_DIR + "hlc_dm2.fits"

    planet_x_offset = 8  # Lambda/D separation
    planet_contrast = 1e0  # Bright companion for clear speckle diagnostic check

    print("--> Simulating Coherent Stellar Wavefront...")
    field_star, sampl = propagate_hlc_system(
        HLC_BAND1_LAMBDA0_M, 2.363114, HLC_BAND1_GRID_SIZE, HLC_BAND1_BEAM_RATIO,
        pupil_path, fpm_real_path, fpm_imag_path, dm1_path, dm2_path, lyot_path,
        is_planet=False
    )
    
    print(f"--> Simulating Off-Axis Companion Wavefront at {planet_x_offset} L/D...")
    field_planet, _ = propagate_hlc_system(
        HLC_BAND1_LAMBDA0_M, 2.363114, HLC_BAND1_GRID_SIZE, HLC_BAND1_BEAM_RATIO,
        pupil_path, fpm_real_path, fpm_imag_path, dm1_path, dm2_path, lyot_path,
        offset_x_lamD=planet_x_offset, is_planet=True, planet_contrast=planet_contrast
    )

    # Coherent combination of focused electric fields onto the detector matrix
    combined_intensity = np.abs(field_star)**2 + np.abs(field_planet)**2
    norm_intensity = combined_intensity / np.max(np.abs(field_star)**2)

    # Establish accurate Lambda/D coordinate boundary framework for presentation plotting
    out_dim = combined_intensity.shape[0]
    extent_limit = (out_dim / 2.0) * 0.1  # 0.1 lambda_0/D per pixel configuration scaling

    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(norm_intensity + 1e-10), origin="lower", 
               extent=[-extent_limit, extent_limit, -extent_limit, extent_limit], cmap='magma', vmin=-10, vmax=0)
    
    # Trace target working bounds (3 to 9 Lambda/D) to track Dark Hole symmetry explicitly
    ax = plt.gca()
    ax.add_patch(plt.Circle((0, 0), 3, color='w', fill=False, ls=':', alpha=0.6, label='Dark Hole Boundaries'))
    ax.add_patch(plt.Circle((0, 0), 9, color='w', fill=False, ls=':', alpha=0.6))
    
    plt.title(f"Coherent Roman HLC Simulation (Planet at {planet_x_offset} $\\lambda_0/D$)", fontsize=14)
    plt.xlabel("Angular Separation [$\\lambda_0/D$]", fontsize=12)
    plt.ylabel("Angular Separation [$\\lambda_0/D$]", fontsize=12)
    plt.colorbar(label="Log10 Relative Intensity")
    plt.show()