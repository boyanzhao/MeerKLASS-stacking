#%% 并行优化版：预计算几何映射 + phi 稀疏采样积分 + HEALPix 输出
print("start convolution")
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import time
import sys
import h5py
from astropy.io import fits
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
from joblib import Parallel, delayed

start_time = time.time()

# =========================
# 参数
# =========================
N_JOBS = max(1, os.cpu_count() // 2)
Nside = 256

# 常改参数
theta_max          = 90
beam_rotation      = 0
sky_component_list = ["f1", "s1", "d1", "c1"]
N_phi = 360/3

# 不常改参数
Ra_min_deg, Ra_max_deg = 330.0, 360.0
Dec_min_deg, Dec_max_deg = -36, -25

theta_start_deg = 0.0
phi_start_deg = 0.0
phi_range_deg = 360.0

beam_file_path = "../data/beam.h5"

# 将 sky component 列表拼接为字符串
sky_components_str = "".join(sky_component_list)

output_path = (
    "../healpix_map/"
    + "dirk_healpix_N"
    +str(N_phi)
    +"_"
    + "sp"+str(int(beam_rotation))
    + "_theta"+str(int(theta_max))
    + "_"+sky_components_str
    + ".fits"
)

print(output_path)
print("N_phi = "+str(N_phi))

# =========================
# 波束phi旋转
# =========================
def rotate_beam_phi_deg(beam_cube, phi_grid_deg, rotation_deg):
    """沿phi方向旋转波束（周期插值）"""
    phi = np.asarray(phi_grid_deg)
    beam = np.asarray(beam_cube)

    idx = np.argsort(phi)
    phi_sorted = phi[idx]
    beam_sorted = beam[:, idx, :]

    phi_ext = np.concatenate([phi_sorted-360, phi_sorted, phi_sorted+360])
    beam_ext = np.concatenate([beam_sorted]*3, axis=1)

    phi_src = np.mod(phi - rotation_deg, 360)
    out = np.empty_like(beam)

    for f in range(beam.shape[0]):
        for t in range(beam.shape[2]):
            out[f, :, t] = np.interp(phi_src, phi_ext, beam_ext[f, :, t])
    return out


# =========================
# 固定北极参考坐标系（带终止机制）
# =========================
def build_local_basis(nside, center_pix):
    """
    固定北极为参考方向。
    若方向接近南北极（数值不稳定），直接终止程序。
    """

    z_axis = np.array(hp.pix2vec(nside, center_pix))
    z_axis /= np.linalg.norm(z_axis)

    ref_vec = np.array([0.0, 0.0, 1.0])

    # ======== 关键修改：检测并终止 ========
    if abs(np.dot(ref_vec, z_axis)) > 0.999:
        print("错误: 太靠近南北极")
        sys.exit(1)

    y_axis = np.cross(z_axis, ref_vec)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    return x_axis, y_axis, z_axis


# =========================
# phi采样
# =========================
def build_phi_samples(phi_min, phi_range, n):
    return np.linspace(phi_min, phi_min+phi_range, n, endpoint=False)


def find_nearest_phi_indices(phi_grid, phi_samples):
    return np.array([np.argmin(abs(phi_grid - v)) for v in phi_samples])


# =========================
# 几何预计算
# =========================
def precompute_pixel_geometry(
    nside, pix,
    theta_grid, phi_grid,
    theta_min, theta_max,
    phi_min, phi_range,
    n_phi_sample,
):

    x_axis, y_axis, z_axis = build_local_basis(nside, pix)

    theta_idx = np.where(
        (theta_grid >= theta_min) &
        (theta_grid <= theta_max)
    )[0]

    phi_samples = build_phi_samples(phi_min, phi_range, n_phi_sample)
    phi_idx = np.unique(find_nearest_phi_indices(phi_grid, phi_samples))

    theta = np.deg2rad(theta_grid[theta_idx])
    phi = np.deg2rad(phi_grid[phi_idx])

    Phi, Theta = np.meshgrid(phi, theta, indexing="ij")

    x = np.sin(Theta)*np.cos(Phi)
    y = np.sin(Theta)*np.sin(Phi)
    z = np.cos(Theta)

    vec = (
        x[...,None]*x_axis +
        y[...,None]*y_axis +
        z[...,None]*z_axis
    ).reshape(-1,3)

    theta_o = np.arccos(vec[:,2])
    phi_o = np.mod(np.arctan2(vec[:,1], vec[:,0]), 2*np.pi)

    d_theta = np.deg2rad(0.1)
    d_phi = np.deg2rad(0.1)

    dS = (np.sin(np.deg2rad(theta_grid[theta_idx])) * d_theta * d_phi)[None,:]

    return {
        "theta_orig": theta_o,
        "phi_orig": phi_o,
        "theta_idx": theta_idx,
        "phi_idx": phi_idx,
        "shape": (phi_idx.size, theta_idx.size),
        "dS": dS,
        "patch_solid_angle": n_phi_sample/3600 * 4*np.pi,
    }


# =========================
# 卷积
# =========================
def convolve_one_pixel(geom, hpmap, beam_patch):
    sky = hp.get_interp_val(
        hpmap,
        geom["theta_orig"],
        geom["phi_orig"]
    ).reshape(geom["shape"])

    return np.sum(beam_patch * sky * geom["dS"]) / geom["patch_solid_angle"]


def process_one_freq(i, fval, beam_cube, geom_cache, sky_model):

    beam = beam_cube[i]

    sky = sky_model.get_emission(fval * u.MHz)[0]
    sky = sky.to(u.mK_CMB, equivalencies=u.cmb_equivalencies(fval*u.MHz)).value

    res = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(convolve_one_pixel)(
            g,
            sky,
            beam[np.ix_(g["phi_idx"], g["theta_idx"])]
        )
        for g in geom_cache
    )

    return np.array(res)


# =========================
# 主程序
# =========================

# 读取波束
with h5py.File(beam_file_path, "r") as f:
    beam_cube = f["data"][:]
    freq = f["frequency"][:]
    beam_phi = f["phi"][:]
    beam_theta = f["theta"][:]

# 旋转波束
beam_cube = rotate_beam_phi_deg(beam_cube, beam_phi, beam_rotation)

# 天空模型
sky_model = pysm3.Sky(nside=Nside, preset_strings=sky_component_list)

# 选区域
npix = hp.nside2npix(Nside)
ipix = np.arange(npix)

theta, phi = hp.pix2ang(Nside, ipix)
ra = phi
dec = np.pi/2 - theta

mask = (
    (ra > np.deg2rad(Ra_min_deg)) &
    (ra < np.deg2rad(Ra_max_deg)) &
    (dec > np.deg2rad(Dec_min_deg)) &
    (dec < np.deg2rad(Dec_max_deg))
)

selected = ipix[mask]

# 几何缓存
geom_cache = [
    precompute_pixel_geometry(
        Nside, p,
        beam_theta, beam_phi,
        theta_start_deg,
        theta_start_deg + theta_max,
        phi_start_deg,
        phi_range_deg,
        N_phi,
    )
    for p in selected
]

# 主循环
result = np.zeros((len(selected), len(freq)))

for i, fval in enumerate(freq):
    print(f"{i+1}/{len(freq)} {fval:.2f} MHz")
    result[:, i] = process_one_freq(i, fval, beam_cube, geom_cache, sky_model)

# 还原HEALPix
npix_full = hp.nside2npix(Nside)
cube = np.full((len(freq), npix_full), hp.UNSEEN)

for i in range(len(freq)):
    cube[i, selected] = result[:, i]

# 保存
Column_names = [
    f"F{float(fval):.3f}MHz"
    for fval in freq
]
hp.write_map(
    output_path,
    cube,
    column_names=Column_names,
    overwrite=True,
    dtype=np.float64,
)

print("DONE")
print("Total time:", (time.time()-start_time)/60, "min")