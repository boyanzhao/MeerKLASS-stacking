import time
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from mpi4py import MPI
from config import *
from meer21cm import PowerSpectrum, Specification
from meer21cm.mock import HIGalaxySimulation
from meer21cm.plot import plot_map
from meer21cm.stack import stack, sum_3d_stack
from meer21cm.util import center_to_edges, jy_to_kelvin, pca_clean, rebin_spectrum
START_TIME = time.time()
print("start shuffle dirk")

# 参数设置
beam_rotation = 180
N_phi = 75

Use_counts = True
Use_poisson = False

sky_components = "f1s1d1c1"
theta_range = 90

# 文件路径
fits_file = (
    f"../wcs_map/"
    f"dirk_wcs_N{N_phi}"
    f"_sp{beam_rotation}_"
    f"theta{theta_range}_"
    f"{sky_components}.fits"
)
save_dir = "../dirk_results/"

save_file_name = (
    f"N{N_phi}_"
    f"sp{beam_rotation}_"
    f"theta{theta_range}_"
    f"{sky_components}"
)
save_file_name += "_counts_3D" if Use_counts else "_uniform_3D"
if Use_poisson:
    save_file_name += "_Poisson"
print(f"保存位置: {save_dir}{save_file_name}.npy")

# 初始化 HI 模拟与规格对象
ps = HIGalaxySimulation(
    map_file=fits_file,
    counts_file=counts_file,
    cosmo=cosmo,
    gal_file=gal_file,
    ra_range=ra_range_MK,
    dec_range=dec_range_MK,
    survey='meerklass_2021', 
    band='L'
)
sp_temp = Specification(
    map_file=fits_file,
    counts_file=counts_file,
    cosmo=cosmo,
    gal_file=gal_file,
    ra_range=ra_range_GAMA,
    dec_range=dec_range_GAMA,
    survey='meerklass_2021', 
    band='L'
)

# 读取数据（HI map 与星系 catalog）
ps.read_from_fits()
ps.read_gal_cat()

# 添加噪声
if Use_counts:
    NoiseStd = 17/np.sqrt(2*2*0.2*1e6) # Tsys = 16 MeerKLASS 2025
    noise = ps.create_white_noise_map(NoiseStd,counts=ps.counts, seed=42, inf_to_zero=True)
else:
    Counts   = np.where(ps.counts == 0, np.nan, ps.counts) # replace 0 with Nan so that they won't participate in np.nanmean
    NoiseStd = np.nanmean(17/np.sqrt(2*2*0.2*1e6*Counts)) # Eq (20) in MeerKLASS 2025
    noise = ps.create_white_noise_map(NoiseStd,counts=None, seed=42, inf_to_zero=True)  
    print("uniform noise level = " +str(NoiseStd)+"K")
ps.data += noise

# 使用 GAMA catalog 替换 ps 内部的星系信息
sp_temp.read_gal_cat()
ps._ra_gal  = sp_temp.ra_gal
ps._dec_gal = sp_temp.dec_gal
ps._z_gal   = sp_temp.z_gal

# 单位转换：Jy -> Kelvin
jy_k_coeff = jy_to_kelvin(
    1,
    ps.pixel_area * (np.pi / 180) ** 2,
    ps.nu
)

print("NaN in ps.data before PCA:", np.isnan(ps.data).sum())
print("Inf in ps.data before PCA:", np.isinf(ps.data).sum())

# PCA 前景清除
hi_map_clean_data, A_mat_data = pca_clean(
    signal = ps.data / jy_k_coeff[None, None],
    N_fg = 10,
    weights=ps.W_HI,
    mean_center=True,
    return_A=True,
    ignore_nan = False
)
R_mat_MK = np.eye(len(ps.nu)) - A_mat_data @ A_mat_data.T              # projector
ps._data = hi_map_clean_data                                           

# 3D stacking
stack_3D_map, stack_3D_weight = stack(
    ps,
    symmetrize=symmetrize,
)

# 角向与谱向压缩
angular_stack_osci_data, spectral_stack_osci_data = sum_3d_stack(
    stack_3D_map,
    ang_sum_dist=1.0 / 0.3
)

# 谱向 rebin
spectral_stack_rebin_osci_data = rebin_spectrum( spectral_stack_osci_data )

# 构造 bin（角度 & 速度）
ang_edges = np.linspace(-10, 10, 21) * ps.pix_resol # 角向 bin 边界

x_edges = np.linspace(0, 2 * ps.nu.size, 2 * ps.nu.size + 1) * ps.vel_resol # 速度方向 bin（以中心对齐）
x_edges -= x_edges[x_edges.size // 2]
x_edges = center_to_edges(x_edges)

vel_bin = (x_edges[1:] + x_edges[:-1]) / 2 # bin 中心
x_rebin = center_to_edges(rebin_spectrum(vel_bin)) # rebin 后的边界

# 记录星系数量 & 权重
num_g_in_GAMA = ps.ra_gal.size
W_HI = ps.W_HI.copy()
w_HI = ps.w_HI.copy()

# 单次随机样本 stacking 函数
def one_random_sample(seed):
    """
    生成一个随机 HI + galaxy mock，并进行 stacking
    """

    # --- 构建 mock HI + tracer ---
    hisim = HIGalaxySimulation(
        use_poisson = Use_poisson,
        ra_range=ra_range_GAMA,
        dec_range=dec_range_GAMA,
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        num_discrete_source=int(num_g_in_GAMA),
        seed=seed,
        downres_factor_radial=1 / 2,
        downres_factor_transverse=1 / 2,
        target_relative_to_num_g=1.5,
        kmax=15,
        nonlinear='both',
        tf_slope=3.66,
        tf_zero=1.6,
        no_vel=False,
        highres_sim=3,
        strict_num_source=True,
        survey='meerklass_2021', 
        band='L'
    )

    # --- 设置模拟盒 ---
    hisim.get_enclosing_box()

    # 调整目标密度
    hisim.target_relative_to_num_g = (
        np.prod(hisim.box_len) / hisim.survey_volume * 1.2
    )

    # 将 mock tracer 投影为 galaxy catalog
    hisim.propagate_mock_tracer_to_gal_cat()

    # --- 构造 stacking 用 Specification ---
    sp = Specification(
        map_file=fits_file,
        counts_file=counts_file,
        cosmo=cosmo,
        gal_file=gal_file,
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        survey='meerklass_2021', 
        band='L'
    )
    # 使用真实 cleaned HI map
    sp._data = hi_map_clean_data.copy()                             
    sp.W_HI = W_HI.copy()
    sp.w_HI = w_HI.copy()
    # 使用 mock galaxy
    sp._ra_gal = hisim.ra_gal
    sp._dec_gal = hisim.dec_gal
    sp._z_gal = hisim.z_gal
    # stacking
    stack_3D_map, stack_3D_weight = stack(
        sp,
        symmetrize=symmetrize,
    )
    return stack_3D_map

# MPI 并行部分
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()
print("len(seed_list) = " + str(len(seed_list)))

stack_3D_rand = []

# 按 seed 分配任务到不同 rank
for idx, seed_i in enumerate(seed_list):
    if seed_i % size == rank:
        # 计算进度百分比（浮点数格式化输出）
        progress = idx / len(seed_list) * 100
        print(f"{progress:.2f}%")
        
        stack_3D_i = one_random_sample(seed_i)
        stack_3D_rand.append(stack_3D_i)

stack_3D_rand =np.array(stack_3D_rand)
# 每个 rank 保存自己的结果

# save_file_name += f"_rand_{rank}"
np.save(save_dir + save_file_name, stack_3D_rand)
print("已保存")
comm.Barrier()

END_TIME = time.time()
TOTAL_TIME = END_TIME - START_TIME
hours = int(TOTAL_TIME // 3600)
minutes = int((TOTAL_TIME % 3600) // 60)
seconds = TOTAL_TIME % 60
print(
    f"[Rank {rank}] 总运行时间: "
    f"{hours:02d}h "
    f"{minutes:02d}m "
    f"{seconds:05.2f}s",
    flush=True
)

# （可选）汇总所有 rank 结果

# if rank == 0:
#     stack_3D_rand = []
#     for i in range(size):
#         stack_3D_rand += [
#             np.load(save_dir + f'stack_3D_rand_{i}.npy'),
#         ]
#     stack_3D_rand = np.concatenate(*stack_3D_rand, axis=0)
#     np.save(save_dir + 'stack_3D_rand', stack_3D_rand)



# （备用）多进程版本（非 MPI）

# if __name__ == "__main__":
#     stack_rand_arr = []
#     with Pool() as pool:
#         for map_i in pool.map(
#             one_random_sample,
#             seed_list
#         ):
#             stack_rand_arr += [map_i]
#
#     stack_rand_arr = np.array(stack_rand_arr)
#     np.save(save_dir + 'stack_random_pos', stack_rand_arr)