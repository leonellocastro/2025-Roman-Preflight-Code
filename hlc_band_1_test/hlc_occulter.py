import proper
import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits

def hlc_occulter(wavelength, diam, scale_occulter, grid_size, beam_ratio, f_lens, pupil, fpm_real, fpm_imag, dm1, dm2, lyot_stop):
    # 1. Initialize the wavefront at the entrance pupil
    wfo = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)

    # 2. Read the pupil map and apply it to the wavefront
    sampling = diam/grid_size
    pupil = proper.prop_errormap( wfo, pupil, 0, 0, AMPLITUDE=True, SAMPLING=sampling)
    print("Type of pupil:", type(pupil), "Shape of pupil:", pupil.shape)
    proper.prop_multiply(wfo, pupil)
    amplitude = proper.prop_get_amplitude(wfo)

    # 3. Plot the entrance pupil
    plt.figure(figsize=(12,8))
    plt.imshow(amplitude, origin = "lower", cmap = plt.cm.gray)
    plt.title("Entrance Pupil", fontsize = 18)
    plt.show()

    dm1 = fits.getdata(dm1)
    dm2 = fits.getdata(dm2)
    proper.prop_dm(wfo, dm1, 0.0, 0.0, 0.2, n_actuators=48)
    proper.prop_propagate(wfo, 1.0, "DM2")
    proper.prop_dm(wfo, dm2, 0.0, 0.0, 0.2, n_actuators=48)

    # 4. Plot the pupil plane after DMs
    plt.figure(figsize=(12,8))
    amplitude_after_dms = proper.prop_get_amplitude(wfo)
    plt.imshow(amplitude_after_dms, origin = "lower", cmap = plt.cm.gray)
    plt.title("Pupil Plane - After DMs", fontsize = 18)
    plt.show()

    # 5. Propagate to the focal plane (F1) to see the effect of DMs before the occulter
    proper.prop_lens(wfo, f_lens, "Focal Plane after DMs")
    proper.prop_propagate(wfo, f_lens, "Focal Plane after DMs")
    plt.figure(figsize=(12,8))
    plt.imshow((proper.prop_get_amplitude(wfo))**(1/4), origin = "lower", cmap = plt.cm.gray)
    plt.title("Focal Plane - After DMs", fontsize = 18)
    plt.show()

    # 6. Multiply by the occulter (if scale_occulter is not zero, otherwise skip this step for the "no occulter" case)
    if (scale_occulter != 0):
        fpm_s = proper.prop_get_sampling(wfo)
        fpm_real_map = proper.prop_readmap(wfo, fpm_real, 0, 0, AMPLITUDE=True, SAMPLING=fpm_s*scale_occulter)
        fpm_imag_map = proper.prop_readmap(wfo, fpm_imag, 0, 0, AMPLITUDE=True, SAMPLING=fpm_s*scale_occulter)
        fpm_real = np.fft.fftshift(fpm_real_map)
        fpm_imag = np.fft.fftshift(fpm_imag_map)
        fpm_complex = fpm_real + 1j * fpm_imag
        proper.prop_multiply(wfo, fpm_complex)
        max_no_occulter_amplitude = np.max(proper.prop_get_amplitude(wfo))

        # 7. Plot the wavefront after the occulter
        plt.figure(figsize=(12,8))
        plt.imshow((proper.prop_get_amplitude(wfo))**(1/4), origin = "lower", cmap = plt.cm.gray)
        plt.title("Focal Plane 1 - After Occulter", fontsize = 18)
        plt.show()
        max_amplitude_occulter = np.max(proper.prop_get_amplitude(wfo))
        occulter_no_occulter = max_amplitude_occulter / max_no_occulter_amplitude
        print(f"Occulter suppression factor (peak amplitude with occulter / peak amplitude without occulter): {occulter_no_occulter:.2e}")

    # 8. Propagate to Lyot Plane
    proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")
    proper.prop_lens(wfo, f_lens, "Lyot Plane - no Lyot Stop")
    proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")

    # 9. Plot the wavefront at the Lyot Plane before Lyot Stop
    plt.figure(figsize=(12,8))
    amplitude_before_lyot = proper.prop_get_amplitude(wfo)
    plt.imshow(amplitude_before_lyot, origin = "lower", cmap = plt.cm.gray)
    plt.title("Lyot Plane - Before Lyot Stop", fontsize = 18)
    plt.show()

    # 10. Apply Lyot Stop using Lyot Stop file
    lyot_map = proper.prop_readmap(wfo, lyot_stop, SAMPLING=proper.prop_get_sampling(wfo))
    
    proper.prop_multiply(wfo, lyot_map)

    # 11. Plot the wavefront at the Lyot Plane after Lyot Stop
    plt.figure(figsize=(12,8))
    amplitude_after_lyot = proper.prop_get_amplitude(wfo)
    amplitude_after_lyot = np.fft.fftshift(amplitude_after_lyot)
    plt.imshow(amplitude_after_lyot, origin = "lower", cmap = plt.cm.gray)
    plt.title("Lyot Plane - After Lyot Stop", fontsize = 18)
    plt.show()

    # 12. Propagate to the final image plane (detector)
    proper.prop_propagate(wfo, f_lens, "Detector")
    proper.prop_lens(wfo, f_lens, "Detector")
    proper.prop_propagate(wfo, f_lens, "Detector")
    amplitude_detector = proper.prop_get_amplitude(wfo)
    plt.figure(figsize=(12,8))
    plt.imshow((amplitude_detector)**(1/4), origin = "lower", cmap = plt.cm.gray)
    plt.title("Final Image Plane (Detector)", fontsize = 18)
    plt.show()

    # 13. End the PROPER simulation
    sampling = proper.prop_get_sampling(wfo)

    # 14. Save the final image plane field to a FITS file for later analysis (if scale_occulter is not zero, otherwise skip this step for the "no occulter" case)
    if (scale_occulter != 0):
        hdu = fits.PrimaryHDU(amplitude_detector)
        hdu.header['MODE'] = 'HLC Band 1'
        hdu.header['ITER'] = 'Final'
        hdu.writeto('hlc_band1_proper_results.fits', overwrite=True)
    return (wfo, sampling)