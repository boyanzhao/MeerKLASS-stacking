from meer21cm import Specification, PowerSpectrum
from meer21cm.mock import HIGalaxySimulation
import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from astropy import constants, units
from meer21cm.util import center_to_edges, pca_clean, jy_to_kelvin, rebin_spectrum
from meer21cm.stack import stack, sum_3d_stack
from mpi4py import MPI
from project_with_R import project_with_R
from config import *

print("checkpoint1")

Use_poisson = False
save_dir = 'real_results/'
save_file_name = f'stack_3D'

if Use_poisson == True:
    save_file_name += "_Poisson"
print("保存位置"+save_dir + save_file_name+".npy")    

# =========================
# 初始化 HI 模拟与规格对象
# =========================
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

print("checkpoint2")

# =========================
# 读取数据（HI map 与星系 catalog）
# =========================
ps.read_from_fits()
ps.read_gal_cat()
sp_temp.read_gal_cat()

# 使用 GAMA catalog 替换 ps 内部的星系信息
ps._ra_gal = sp_temp.ra_gal
ps._dec_gal = sp_temp.dec_gal
ps._z_gal = sp_temp.z_gal

# =========================
# 单位转换：Jy -> Kelvin
# =========================
jy_k_coeff = jy_to_kelvin(
    1,
    ps.pixel_area * (np.pi / 180) ** 2,
    ps.nu
)

print("checkpoint3")

# =========================
# PCA 前景清除
# =========================
hi_map_clean_data, A_mat_data = pca_clean(
    signal = ps.data / jy_k_coeff[None, None],
    N_fg = 10,
    weights=ps.W_HI,
    mean_center=True,
    return_A=True,
    ignore_nan = False
)


R_mat_MK = np.eye(len(ps.nu)) - A_mat_data @ A_mat_data.T              # projector
noise = np.random.normal(loc=0.0, scale=1.2, size=ps.data.shape)       # 1.2 mK noise

mask = (ps.data != 0)                                                  # find pixels that are not observed
noise = noise * mask                                                   # mask pixels that are not observed
noise = noise / jy_k_coeff[None, None]                                 # mK → Jy
noise_clean_map = project_with_R(noise, R_mat_MK)                      # R acts on n
noise_clean_map *= mask                                                # mask pixels that are not observed

# ps._data = hi_map_clean_data                                         #Zhaoting
ps._data = noise_clean_map                                             #Boyan

print("checkpoint4")

# =========================
# 3D stacking
# =========================
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
spectral_stack_rebin_osci_data = rebin_spectrum(
    spectral_stack_osci_data
)


# =========================
# 构造 bin（角度 & 速度）
# =========================
# 角向 bin 边界
ang_edges = np.linspace(-10, 10, 21) * ps.pix_resol

# 速度方向 bin（以中心对齐）
x_edges = np.linspace(0, 2 * ps.nu.size, 2 * ps.nu.size + 1) * ps.vel_resol
x_edges -= x_edges[x_edges.size // 2]
x_edges = center_to_edges(x_edges)

# bin 中心
vel_bin = (x_edges[1:] + x_edges[:-1]) / 2

# rebin 后的边界
x_rebin = center_to_edges(rebin_spectrum(vel_bin))

# =========================
# 保存 stacking 结果
# =========================
np.save(save_dir + 'stack_3D_data_nosym', stack_3D_map)
np.save(save_dir + 'spectral_stack_data_nosym', spectral_stack_rebin_osci_data)
np.save(save_dir + 'angular_stack_data_nosym', angular_stack_osci_data)


print("checkpoint5")


# =========================
# 记录星系数量 & 权重
# =========================
num_g_in_GAMA = ps.ra_gal.size

W_HI = ps.W_HI.copy()
w_HI = ps.w_HI.copy()


# =========================
# 单次随机样本 stacking 函数
# =========================
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
    sp._data = noise_clean_map.copy()                                       # Boyan
    sp.W_HI = W_HI.copy()
    sp.w_HI = w_HI.copy()

    # 使用 mock galaxy
    sp._ra_gal = hisim.ra_gal
    sp._dec_gal = hisim.dec_gal
    sp._z_gal = hisim.z_gal

    # --- stacking ---
    stack_3D_map, stack_3D_weight = stack(
        sp,
        symmetrize=symmetrize,
    )

    return stack_3D_map

print("checkpoint6")

# =========================
# MPI 并行部分
# =========================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()

print("checkpoint7")
print("len(seed_list) = " + str(len(seed_list)))

stack_3D_rand = []

# 按 seed 分配任务到不同 rank
for idx, seed_i in enumerate(seed_list):
    if seed_i % size == rank:
        print(seed_i)
        
        # 计算进度百分比（浮点数格式化输出）
        progress = idx / len(seed_list) * 100
        print(f"{progress:.2f}%")
        
        stack_3D_i = one_random_sample(seed_i)
        stack_3D_rand.append(stack_3D_i)

stack_3D_rand =np.array(stack_3D_rand)
# 每个 rank 保存自己的结果
save_file_name += f"_rand_{rank}"
np.save(save_dir + save_file_name, stack_3D_rand)
print("已保存")
comm.Barrier()

# =========================
# （可选）汇总所有 rank 结果
# =========================
# if rank == 0:
#     stack_3D_rand = []
#     for i in range(size):
#         stack_3D_rand += [
#             np.load(save_dir + f'stack_3D_rand_{i}.npy'),
#         ]
#     stack_3D_rand = np.concatenate(*stack_3D_rand, axis=0)
#     np.save(save_dir + 'stack_3D_rand', stack_3D_rand)


# =========================
# （备用）多进程版本（非 MPI）
# =========================
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