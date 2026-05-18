from meer21cm import Specification, PowerSpectrum
from meer21cm.mock import HIGalaxySimulation
import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from astropy import constants,units
from meer21cm.util import center_to_edges
from meer21cm.util import pcaclean, jy_to_kelvin
from meer21cm.stack import stack,sum_3d_stack
from meer21cm.util import rebin_spectrum, center_to_edges
from config import *
from mpi4py import MPI

ps = HIGalaxySimulation(
    map_file=fits_file,
    counts_file=counts_file,
    cosmo=cosmo,
    gal_file=gal_file,
    ra_range=ra_range_MK,
    dec_range=dec_range_MK,
)
sp_temp = Specification(
    map_file=fits_file,
    counts_file=counts_file,
    cosmo=cosmo,
    gal_file=gal_file,
    ra_range = ra_range_GAMA,
    dec_range = dec_range_GAMA,
)
# read in map_file
ps.read_from_fits()
ps.read_gal_cat()
# read in galaxy file
sp_temp.read_gal_cat()
ps._ra_gal = sp_temp.ra_gal
ps._dec_gal = sp_temp.dec_gal
ps._z_gal = sp_temp.z_gal
# convert to Jy
jy_k_coeff = jy_to_kelvin(1,ps.pixel_area*(np.pi/180)**2,ps.nu)
#PCA cleaning
hi_map_clean_data,A_mat_data = pcaclean(ps.data/jy_k_coeff[None,None],10,weights=ps.W_HI,mean_centre=True,return_A=True)
R_mat_MK = np.eye(len(ps.nu))-A_mat_data@A_mat_data.T
ps._data = hi_map_clean_data
#stacking
stack_3D_map, stack_3D_weight = stack(
    ps,
    symmetrize=symmetrize,
)
angular_stack_osci_data,spectral_stack_osci_data = sum_3d_stack(
        stack_3D_map,ang_sum_dist=1.0/0.3
    )
spectral_stack_rebin_osci_data = rebin_spectrum(spectral_stack_osci_data)

ang_edges = np.linspace(-10,10,21)*ps.pix_resol
x_edges = np.linspace(0,2*ps.nu.size,2*ps.nu.size+1)*ps.vel_resol
x_edges -= x_edges[x_edges.size//2]
x_edges = center_to_edges(x_edges)
# x_bins
vel_bin = (x_edges[1:]+x_edges[:-1])/2
x_rebin = center_to_edges(rebin_spectrum(vel_bin))
#np.save(save_dir+'x_bins',x_rebin)
np.save(save_dir+'stack_3D_data_nosym',stack_3D_map)
np.save(save_dir+'spectral_stack_data_nosym',spectral_stack_rebin_osci_data)
#np.save(save_dir+'ang_edges',ang_edges)
np.save(save_dir+'angular_stack_data_nosym',angular_stack_osci_data)
print('start')
num_g_in_GAMA = ps.ra_gal.size

W_HI = ps.W_HI.copy()
w_HI = ps.w_HI.copy()


def one_random_sample(seed):
    hisim = HIGalaxySimulation(
        ra_range = ra_range_GAMA,
        dec_range = dec_range_GAMA,
        tracer_bias_1 = 1.5,
        tracer_bias_2 = 1.9,
        num_discrete_source = int(num_g_in_GAMA),
        seed=seed,
        downres_factor_radial=1/2,
        downres_factor_transverse=1/2,
        target_relative_to_num_g=1.5,
        kmax=15,
        nonlinear='both',
        tf_slope=3.66,
        tf_zero=1.6,
        no_vel=False,
        highres_sim=3,
        strict_num_source=True,
    )
    hisim.get_enclosing_box()
    hisim.target_relative_to_num_g = np.prod(hisim.box_len)/hisim.survey_volume * 1.2
    hisim.propagate_mock_tracer_to_gal_cat()
    sp = Specification(
        map_file=fits_file,
        counts_file=counts_file,
        cosmo=cosmo,
        gal_file=gal_file,
        ra_range = ra_range_MK,
        dec_range = dec_range_MK,
    )
    sp._data = hi_map_clean_data.copy()
    sp.W_HI = W_HI.copy()
    sp.w_HI = w_HI.copy()
    sp._ra_gal = hisim.ra_gal
    sp._dec_gal = hisim.dec_gal
    sp._z_gal = hisim.z_gal
    stack_3D_map, stack_3D_weight = stack(
        sp,
        symmetrize=symmetrize,
    )
    return stack_3D_map

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
comm.Barrier()
stack_3D_rand = []
for seed_i in seed_list:
    if seed_i % size == rank:
        print(seed_i)
        stack_3D_i = one_random_sample(seed_i)
        stack_3D_rand += [stack_3D_i,]


np.save(save_dir+f'stack_3D_rand_{rank}',stack_3D_rand)

comm.Barrier()

#if rank == 0:
#    stack_3D_rand = []
#    for i in range(size):
#        stack_3D_rand += [np.load(save_dir+f'stack_3D_rand_{i}.npy'),]
#    stack_3D_rand = np.concatenate(*stack_3D_rand,axis=0)
#    np.save(save_dir+'stack_3D_rand',stack_3D_rand)
        

#if __name__ == "__main__":
#    stack_rand_arr = []
#    with Pool() as pool:
#        for map_i in pool.map(
#            one_random_sample,
#            seed_list
#        ):
#            stack_rand_arr += [map_i,]
#    #for seed in seed_list:
#    #    map_i = one_random_sample(seed)
#    #    stack_rand_arr += [map_i,]
#
#    stack_rand_arr = np.array(stack_rand_arr)
#    np.save(save_dir+'stack_random_pos',stack_rand_arr)
