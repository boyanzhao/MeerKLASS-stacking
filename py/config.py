cosmo='Planck18'
# file_dir = '../'
fits_file =   "/idia/projects/meerklass/MEERKLASS-2/Lband_2021/level6/Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits"
counts_file = '/idia/projects/meerklass/MEERKLASS-2/Lband_2021/level6/Nscan961_Npix_count_cube_p0.3d_sigma4.0_iter2.fits'
gal_file = '../data/G23TilingCatv11.fits'

raminGAMA,ramaxGAMA = 339,351
decminGAMA,decmaxGAMA = -35,-30
ra_range_GAMA = (raminGAMA,ramaxGAMA)
dec_range_GAMA = (decminGAMA,decmaxGAMA)
raminMK,ramaxMK = 334,357
decminMK,decmaxMK = -35,-26.5
ra_range_MK = (raminMK,ramaxMK)
dec_range_MK = (decminMK,decmaxMK)

vel_max=6500

seed_list = range(42,542)

symmetrize = False

Use_poisson = False


