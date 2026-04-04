import os

import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits
import proper
from scipy.interpolate import interp1d

from roman_preflight_proper import ffts, mft2, polmap, trim
import roman_preflight_proper as rp

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

def _show_image(data, title, power=1.0, fft_center=False):
    image = np.array(data, copy=True)
    if fft_center:
        image = np.fft.fftshift(image)
    image = np.abs(image)
    if power != 1.0:
        image = np.power(image, power)
    plt.figure(figsize=(12, 8))
    plt.imshow(image, origin="lower", cmap=plt.cm.gray)
    plt.title(title, fontsize=18)
    plt.show()


def _plot_plane(wavefront, title, power=1.0, fft_center=False, plot_planes=True):
    if plot_planes:
        _show_image(proper.prop_get_wavefront(wavefront), title, power=power, fft_center=fft_center)


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


def _to_from_doublet(
    wavefront,
    wavelength_m,
    dz_to_lens,
    dz_from_lens,
    r1_a,
    r2_a,
    thickness_a,
    glass_a,
    separation_m,
    r1_b,
    r2_b,
    thickness_b,
    glass_b,
    surface_name,
    next_surface_name,
    error_map_files=' ',
):
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
    real_map, header = fits.getdata(fpm_real_path, header=True)
    imag_map = fits.getdata(fpm_imag_path)
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


def _write_complex_fits(filename, field):
    hdul = fits.HDUList([
        fits.PrimaryHDU(np.real(field).astype(np.float32)),
        fits.ImageHDU(np.imag(field).astype(np.float32), name="IMAG"),
    ])
    hdul[0].header["EXTNAME"] = "REAL"
    hdul.writeto(filename, overwrite=True)


def hlc(
    wavelength,
    diam,
    scale_occulter,
    grid_size,
    beam_ratio,
    f_lens,
    pupil,
    fpm_real,
    fpm_imag,
    dm1,
    dm2,
    lyot_stop,
    use_field_stop=True,
    plot_planes=True,
    output_dim=256,
    final_sampling_lam0=0.1,
    use_polmap=False,
    polaxis=0,
):
    del f_lens

    if grid_size != HLC_BAND1_GRID_SIZE:
        print(f"Warning: roman_preflight HLC band 1 uses grid_size={HLC_BAND1_GRID_SIZE}, but received {grid_size}.")
    if not np.isclose(beam_ratio, HLC_BAND1_BEAM_RATIO):
        print(f"Warning: roman_preflight HLC band 1 uses beam_ratio={HLC_BAND1_BEAM_RATIO:.9f}, but received {beam_ratio:.9f}.")

    dm_version = rp.dm_version  # string, DM version
    dm_dir = rp.dm_files_dir

    use_cvs = 0                 # use CVS instead of telescope? (1=yes, 0=no)
    cvs_stop_x_shift_m = 0      # shift of CVS entrance pupil mask in meters
    cvs_stop_y_shift_m = 0
    cvs_stop_z_shift_m = 0      # shift of CVS entrance pupil mask along optical axis (+ is downstream)
    cvs_stop_rotation_deg = 0   # rotation of CVS entrance pupil mask in degrees
    small_spc_grid = 0          # set to 1 to use 500 pix across pupil, else 1000 (baseline SPCs only)
    pupil_array = 0             # 2D array containing pupil pattern (overrides default)
    pupil_mask_array = 0        # 2D array containing SPC pupil mask pattern (overrides default)
    fpm_array = 0               # 2D array containing FPM mask pattern (overrides default)
    fpm_mask = 0                # 2D array where 1=FPM pattern defined, 0=substrate
    lyot_stop_array = 0         # 2D array containing Lyot stop mask pattern (overrides default)
    field_stop_array = 0        # 2D array containing field stop mask pattern (overrides default)

    cor_type = 'hlc'            # coronagraph type ('hlc', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'none')
    source_x_offset_mas = 0     # source offset in mas (tilt applied at primary)
    source_y_offset_mas = 0                 
    source_x_offset = 0         # source offset in lambda0_m/D radians (tilt applied at primary)
    source_y_offset = 0                 
    cvs_source_z_offset_m = 0               # additional distance between CVS source and next optic, in meters
    cvs_jitter_mirror_x_offset_mas = 0      # source offset in milliarcsec (tilt applied at CVS jitter mirror)
    cvs_jitter_mirror_y_offset_mas = 0      # 
    cvs_jitter_mirror_x_offset = 0          # source offset in lambda0_m/D radians (tilt applied at CVS jitter mirror) 
    cvs_jitter_mirror_y_offset = 0
    # "polaxis" comes from the function input so the caller can match roman_preflight.py runs.
    use_errors = 1              # use optical surface phase errors? 1 or 0 
    zindex = np.array([0,0])    # array of Zernike polynomial indices
    zval_m = np.array([0,0])    # array of Zernike coefficients (meters RMS WFE)
    sm_despace_m = 0            # secondary mirror despace (meters) 
    use_pupil_defocus = 1       # include pupil defocus
    use_aperture = 0            # use apertures on all optics? 1 or 0
    cgi_x_shift_pupdiam = 0     # X,Y shear of wavefront at FSM (bulk displacement of CGI); normalized relative to pupil diameter
    cgi_y_shift_pupdiam = 0          
    cgi_x_shift_m = 0           # X,Y shear of wavefront at FSM (bulk displacement of CGI) in meters
    cgi_y_shift_m = 0          
    end_at_fsm = 0              # end propagation after propagating to FSM (no FSM errors)
    fsm_x_offset_mas = 0        # offset in focal plane caused by tilt of FSM in mas
    fsm_y_offset_mas = 0         
    fsm_x_offset = 0            # offset in focal plane caused by tilt of FSM in lambda0/D
    fsm_y_offset = 0            
    fcm_z_shift_m = 0          # offset (meters) of focus correction mirror (+ increases path length)
    use_dm1 = 0                 # use DM1? 1 or 0
    use_dm2 = 0                 # use DM2? 1 or 0
    dm_v_quant = 110.0 / 2.**16 # DM DAC voltage quantization resolution 
    dm_sampling_m = 0.9906e-3   # actuator spacing in meters
    dm_temp_c = 26.0
    dm1_v = np.zeros((48,48))
    dm1_xc_act = 23.5           # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm1_yc_act = 23.5              
    dm1_xtilt_deg = 9.65        # effective DM tilt in deg including 9.65 deg actual tilt and pupil ellipticity
    dm1_ytilt_deg = 0 
    dm1_ztilt_deg = 0 
    dm2_v = np.zeros((48,48))
    dm2_xc_act = 23.5 - 0.1     # for 48x48 DM, wavefront centered at actuator intersections: (0,0) = 1st actuator center
    dm2_yc_act = 23.5               
    dm2_xtilt_deg = 9.65 
    dm2_ytilt_deg = 0
    dm2_ztilt_deg = 0
    spam_x_shift_pupdiam = 0    # X,Y shift of wavefront at SPAM; normalized relative to pupil diameter
    spam_y_shift_pupdiam = 0
    spam_x_shift_m = 0          # X,Y shift of wavefront at SPAM in meters
    spam_y_shift_m = 0
    use_pupil_mask = 1          # SPC only: use SPC pupil mask (0 or 1)
    mask_x_shift_pupdiam = 0    # X,Y shear of shaped pupil mask; normalized relative to pupil diameter
    mask_y_shift_pupdiam = 0          
    mask_x_shift_m = 0          # X,Y shear of shaped pupil mask in meters
    mask_y_shift_m = 0          
    mask_rotation_deg = 0
    use_fpm = 1                 # use occulter? 1 or 0
    fpm_x_offset = 0            # FPM x,y offset in lambda0/D
    fpm_y_offset = 0
    fpm_x_offset_m = 0          # FPM x,y offset in meters
    fpm_y_offset_m = 0
    fpm_z_shift_m = 0           # occulter offset in meters along optical axis (+ = away from prior optics)
    pinhole_diam_m = 0          # FPM pinhole diameter in meters
    end_at_fpm_exit_pupil = 0   # return field at FPM exit pupil?
    use_lyot_stop = 1           # use Lyot stop? 1 or 0
    lyot_x_shift_pupdiam = 0    # X,Y shear of Lyot stop mask; normalized relative to pupil diameter
    lyot_y_shift_pupdiam = 0  
    lyot_x_shift_m = 0          # X,Y shear of Lyot stop mask in meters
    lyot_y_shift_m = 0  
    lyot_rotation_deg = 0
    use_field_stop = 1          # use field stop (HLC)? 1 or 0
    field_stop_radius_lam0 = 0  # field stop radius in lambda0/D
    field_stop_x_offset = 0     # field stop offset in lambda0/D
    field_stop_y_offset = 0
    field_stop_x_offset_m = 0   # field stop offset in meters
    field_stop_y_offset_m = 0
    use_pupil_lens = 0          # use pupil imaging lens? 0 or 1
    use_defocus_lens = 0        # use defocusing lens? Options are 1, 2, 3, 4
    end_at_exit_pupil = 0       # return exit pupil corresponding to final image plane
    excam_despace_m = 0         # increase in spacing between final optic and detector
    final_sampling_m = 0        # final sampling in meters (overrides final_sampling_lam0)
    final_sampling_lam0 = final_sampling_lam0     # final sampling in lambda0/D
    output_dim = output_dim    # dimension of output in pixels (overrides output_dim0)
    image_x_offset_m = 0        # shift of image at detector plane in meters
    image_y_offset_m = 0


    # more setting
    sm_despace_m = 0.
    diam = 2.363114
    fl_pri = 2.838279206904720
    d_pri_sec = 2.285150508110035 + sm_despace_m
    fl_sec = -0.654200796568004
    diam_sec = 0.58166
    d_sec_pomafold = 2.993753469304728 + sm_despace_m
    diam_pomafold = 0.09
    d_pomafold_m3 = 1.680935841598811
    fl_m3 = 0.430216463069001
    diam_m3 = 0.2
    d_m3_pupil = 0.469156807765176
    d_m3_m4 = 0.943514749358944
    fl_m4 = 0.116239114833590
    diam_m4 = 0.07
    d_m4_m5 = 0.429145636743193
    fl_m5 = 0.198821518772608
    d_m5_pupil = 0.716529242927776
    diam_m5 = 0.07
    d_m5_ttfold = 0.351125431220770
    diam_ttfold = 0.06
    d_ttfold_fsm = d_m5_pupil - d_m5_ttfold
    if use_pupil_defocus:
        d_ttfold_fsm = d_ttfold_fsm + 0.033609

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
    d_oap5_fpm = fl_oap5
    fpm_index = _glass_index("SILICA", wavelength)
    fpm_thickness = 0.006363747896388863
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
    filter_index = _glass_index("SILICA", wavelength)
    filter_thickness = 0.004016105782012525
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095
    d_lens_fold4 = 0.202226
    d_fold4_image = 0.050206330646919


    # initialize wavefront
    wavefront = proper.prop_begin(diam, wavelength, grid_size, beam_ratio) # same now 

    # loading pupil mask, scale and apply to wavefront
    pupil_map = trim(fits.getdata(pupil), grid_size)
    proper.prop_multiply(wavefront, pupil_map)
    proper.prop_define_entrance(wavefront)


    # Define entrance pupil - correct - followed roman preflight
    _plot_plane(wavefront, "Entrance Pupil", plot_planes=plot_planes)

    proper.prop_lens(wavefront, fl_pri)
    if use_polmap and polaxis != 0:
        polmap(wavefront, rp.polfile, HLC_BAND1_PUPIL_DIAM_PIX, polaxis)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True)

    # skip polarization, but keep the rest of the OTA sequence consistent with roman_preflight.py

    proper.prop_propagate(wavefront, d_pri_sec, 'secondary')
    proper.prop_lens(wavefront, fl_sec)

    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits', WAVEFRONT=True)


    proper.prop_propagate(wavefront, d_sec_pomafold, 'POMA FOLD')
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_POMAFOLD_measured_phase_error_V2.0.fits', WAVEFRONT=True)

    _plot_plane(wavefront, "POMA FOLD", plot_planes=plot_planes)


    proper.prop_propagate(wavefront, d_pomafold_m3, 'M3')
    proper.prop_lens(wavefront, fl_m3)

    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M3_measured_phase_error_V2.0.fits', WAVEFRONT=True)

    _plot_plane(wavefront, "M3", plot_planes=plot_planes)


    proper.prop_propagate(wavefront, d_m3_m4, 'M4')
    proper.prop_lens(wavefront, fl_m4)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M4_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "M4", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_m4_m5, 'M5')
    proper.prop_lens(wavefront, fl_m5)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_M5_measured_phase_error_V2.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "M5", plot_planes=plot_planes)


    proper.prop_propagate(wavefront, d_m5_ttfold, 'TT FOLD')
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "TT FOLD", plot_planes=plot_planes)


    proper.prop_propagate(wavefront, d_ttfold_fsm, 'FSM')
    # proper.prop_errormap( wavefront, map_dir+'roman_phasec_LOWORDER_phase_error_V2.0.fits', WAVEFRONT=True )

    proper.prop_errormap( wavefront, map_dir+'roman_phasec_FSM_FLIGHT_coated_measured_phase_error_V3.0.fits', WAVEFRONT=True )

    _plot_plane(wavefront, "FSM", plot_planes=plot_planes)

    # OAP1 
    proper.prop_propagate(wavefront, d_fsm_oap1, "OAP1")
    proper.prop_lens(wavefront, fl_oap1)
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP1_phase_error_V3.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "OAP1", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_oap1_fcm, "FCM")
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "FCM", plot_planes=plot_planes)

    # OPA2
    proper.prop_propagate(wavefront, d_fcm_oap2, "OAP2")
    proper.prop_lens(wavefront, fl_oap2)
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP2_phase_error_V3.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "OAP2", plot_planes=plot_planes)

    # DM1 
    dm_struct = rp.load_cgi_dm_files(dm_files_dir=dm_dir, version=dm_version, temp_c=dm_temp_c)
    dm1_map = proper.prop_fits_read(dm1)
    dm2_map = proper.prop_fits_read(dm2)

    proper.prop_propagate(wavefront, d_oap2_dm1, "DM1")
    _write_complex_fits("hlc_test_dm1.fits", proper.prop_get_wavefront(wavefront))
    
    rp.cgi_dm(
        wavefront, dm_struct, 1, dm1_map, dm_sampling_m=dm_sampling_m, dm_v_quant=dm_v_quant, dm_xc_act=dm1_xc_act, dm_yc_act=dm1_yc_act, dm_xtilt_deg=dm1_xtilt_deg, dm_ytilt_deg=dm1_ytilt_deg, dm_ztilt_deg=dm1_ztilt_deg 
        )
        
    _plot_plane(wavefront, "DM1", plot_planes=plot_planes)
    _write_complex_fits("hlc_test_dm1_abr.fits", proper.prop_get_wavefront(wavefront))
    # DM2
    proper.prop_propagate(wavefront, d_dm1_dm2, "DM2")
    _write_complex_fits("hlc_test_dm2.fits", proper.prop_get_wavefront(wavefront))

    rp.cgi_dm( 
        wavefront, dm_struct, 2, dm2_map, dm_sampling_m=dm_sampling_m, dm_v_quant=dm_v_quant,
        dm_xc_act=dm2_xc_act, dm_yc_act=dm2_yc_act, 
        dm_xtilt_deg=dm2_xtilt_deg, dm_ytilt_deg=dm2_ytilt_deg, dm_ztilt_deg=dm2_ztilt_deg)

    _plot_plane(wavefront, "DM2", plot_planes=plot_planes)
    _write_complex_fits("hlc_test_dm2_abr.fits", proper.prop_get_wavefront(wavefront))

    # OAP3
    proper.prop_propagate(wavefront, d_dm2_oap3, "OAP3")
    proper.prop_lens(wavefront, fl_oap3)
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP3_phase_error_V3.0.fits', WAVEFRONT=True)

    _plot_plane(wavefront, "OAP3", plot_planes=plot_planes)
    
    # FOLD_3
    proper.prop_propagate(wavefront, d_oap3_fold3, "FOLD_3")
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "FOLD_3", plot_planes=plot_planes)
    _write_complex_fits("hlc_test_fold3.fits", proper.prop_get_wavefront(wavefront))

    #OAP4
    proper.prop_propagate(wavefront, d_fold3_oap4, "OAP4")
    proper.prop_lens(wavefront, fl_oap4)
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP4_phase_error_V3.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "OAP4", plot_planes=plot_planes)

    # Pupil Mask
    proper.prop_propagate(wavefront, d_oap4_pupilmask, "PUPIL_MASK")
    proper.prop_errormap(wavefront, map_dir+'roman_phasec_PUPILFOLD_phase_error_V1.0.fits', WAVEFRONT=True )
    _plot_plane(wavefront, "PUPIL_MASK", plot_planes=plot_planes)

    # OAP5
    proper.prop_propagate(wavefront, d_pupilmask_oap5, "OAP5")
    proper.prop_lens(wavefront, fl_oap5)
    proper.prop_errormap( wavefront, map_dir+'roman_phasec_OAP5_phase_error_V3.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "OAP5", plot_planes=plot_planes)

    # FPM - BUG
    proper.prop_propagate(wavefront, d_oap5_fpm+fpm_z_shift_m, "FPM", TO_PLANE=True)

    _write_complex_fits("hlc_test_fpm.fits", proper.prop_get_wavefront(wavefront))

    _plot_plane(wavefront, "FPM - Before Occulter", power=0.25, plot_planes=plot_planes)



    if scale_occulter != 0:
        _apply_hlc_fpm(wavefront, wavelength, fpm_real, fpm_imag)
    _plot_plane(wavefront, "FPM - After Occulter", power=0.25, plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_fpm_oap6, "OAP6")
    proper.prop_lens(wavefront, fl_oap6)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP6_phase_error_V3.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "OAP6", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_oap6_lyotstop, "LYOT STOP")
    _plot_plane(wavefront, "Lyot Plane - Before Lyot Stop", plot_planes=plot_planes)
    lyot_map = trim(fits.getdata(lyot_stop), grid_size)
    proper.prop_multiply(wavefront, lyot_map)
    _plot_plane(wavefront, "Lyot Plane - After Lyot Stop", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_lyotstop_oap7, "OAP7")
    proper.prop_lens(wavefront, fl_oap7)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP7_phase_error_V4.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "OAP7", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_oap7_fieldstop, "FIELD_STOP", TO_PLANE=1)
    _plot_plane(wavefront, "Field Stop - Before Mask", power=0.25, plot_planes=plot_planes)
    if use_field_stop:
        stop_radius_m = HLC_BAND1_FIELD_STOP_RADIUS_LAM0 * (
            HLC_BAND1_M_PER_LAMD_575NM_AT_FIELD_STOP * HLC_BAND1_LAMBDA0_M / 575e-9
        )
        proper.prop_circular_aperture(wavefront, stop_radius_m)
    _plot_plane(wavefront, "Field Stop - After Mask", power=0.25, plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_fieldstop_oap8, "OAP8")
    proper.prop_lens(wavefront, fl_oap8)
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_OAP8_phase_error_V3.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "OAP8", plot_planes=plot_planes)

    proper.prop_propagate(wavefront, d_oap8_filter, "FILTER")
    proper.prop_errormap(wavefront, map_dir + 'roman_phasec_FILTER_phase_error_V1.0.fits', WAVEFRONT=True)
    _plot_plane(wavefront, "FILTER", plot_planes=plot_planes)

    imaging_lens_error_files = [map_dir + 'roman_phasec_LENS_phase_error_V1.0.fits', ' ']

    # taken from roman_preflight.py
    _to_from_doublet(
        wavefront,
        wavelength,
        d_filter_lens,
        d_lens_fold4 + d_fold4_image,
        0.10792660718579995,
        -0.10792660718579995,
        0.003,
        "S-BSL7R",
        0.0005,
        1e10,
        0.10608379812011390,
        0.0025,
        "PBM2R",
        "IMAGING LENS",
        "IMAGE",
        error_map_files=imaging_lens_error_files,
    )

    rsqr = proper.prop_radius(wavefront) ** 2
    field, sampling = proper.prop_end(wavefront, NOABS=True)

    lam_m = np.array([500, 525, 550, 575, 600, 625, 650, 700, 730, 775, 825, 880]) * 1e-9
    cc = np.array([1.1797, 1.1840, 1.1865, 1.1881, 1.1890, 1.1891, 1.1887, 1.1868, 1.1852, 1.1820, 1.1778, 1.1726])
    interp = interp1d(lam_m, cc, kind="linear", fill_value="extrapolate")
    c = interp(wavelength)
    field *= np.exp((1j * np.pi / wavelength * c) * rsqr) * np.exp(-1j * 0.1)

    field = _apply_detector_orientation(field)
    mag = (HLC_BAND1_PUPIL_DIAM_PIX / grid_size) / final_sampling_lam0 * (wavelength / HLC_BAND1_LAMBDA0_M)
    sampling = sampling / mag
    field = proper.prop_magnify(field, mag, output_dim, AMP_CONSERVE=True)

    intensity_detector = np.abs(field) ** 2
    if plot_planes:
        _show_image(intensity_detector, "Final Image Plane (Detector)", power=0.25)

    if scale_occulter != 0:
        hdu = fits.PrimaryHDU(intensity_detector)
        hdu.header["MODE"] = "HLC Band 1"
        hdu.header["ITER"] = "Final"
        hdu.header["SAMPLING"] = sampling
        hdu.writeto("hlc_band1_proper_results.fits", overwrite=True)

    return field, sampling
