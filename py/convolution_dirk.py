#%% 并行优化版：预计算几何映射 + phi 稀疏采样积分 + HEALPix 输出
import os
# =========================
# 限制底层数值库线程数
# =========================
# 这些环境变量应尽量在 numpy 等数值库导入之前设置。
# 目的是避免 joblib 并行时，底层 BLAS / OpenMP 再开很多线程。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import time
import warnings
import h5py
import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
from   joblib import Parallel, delayed
warnings.filterwarnings("ignore")
# =========================
# 参数
# =========================

# 并行参数
N_JOBS = max(1, os.cpu_count() // 2)

# HEALPix 参数
Nside = 256

# 常改参数
beam_rotation      = 60
sky_component_list = ["f1", "s1", "d1", "c1"]
N_phi              = 72


# 是否额外加入 CMB 平均温度
# 注意：代码中 sky 会先被转换到 mK_CMB。
# 因此 2.725 K 对应 2725.0 mK_CMB。
ADD_CMB_MONOPOLE   = False
CMB_MONOPOLE_MK    = 2725.0

# 天区范围，单位是度
Ra_min_deg,  Ra_max_deg  = 330.0, 360.0
Dec_min_deg, Dec_max_deg = -36, -25


# 波束积分范围，单位是度
theta_start_deg = 0.0
theta_max = 90.0
phi_start_deg   = 0.0
phi_max_deg   = 360.0

# 输入文件
beam_file_path  = "../data/beam.h5"

# 输出目录
output_dir      = "../healpix_map/"


# =========================
# 输出文件名
# =========================
def build_output_path():
    """根据当前参数生成输出文件路径。"""

    sky_components_str = "".join(sky_component_list)
    cmb_suffix = "_2725" if ADD_CMB_MONOPOLE else ""

    output_path = (
        output_dir
        + "dirk_healpix_N"
        + str(N_phi)
        + "_"
        + "sp"
        + str(int(beam_rotation))
        + "_theta"
        + str(int(theta_max))
        + "_"
        + sky_components_str
        + cmb_suffix
        + ".fits"
    )

    return output_path


# =========================
# 波束 phi 旋转
# =========================
def rotate_beam_phi_deg(beam_cube, phi_grid_deg, rotation_deg):
    """
    沿 phi 方向旋转波束。

    参数
    ----
    beam_cube:
        波束数组，形状为 (N_freq, N_phi, N_theta)。

    phi_grid_deg:
        phi 网格，单位是度。

    rotation_deg:
        phi 旋转角，单位是度。

    返回
    ----
    out:
        旋转后的波束数组，形状与 beam_cube 相同。
    """

    phi = np.asarray(phi_grid_deg)
    beam = np.asarray(beam_cube)

    idx = np.argsort(phi)
    phi_sorted = phi[idx]
    beam_sorted = beam[:, idx, :]

    phi_ext = np.concatenate(
        [
            phi_sorted - 360,
            phi_sorted,
            phi_sorted + 360,
        ]
    )

    beam_ext = np.concatenate(
        [
            beam_sorted,
            beam_sorted,
            beam_sorted,
        ],
        axis=1,
    )

    phi_src = np.mod(phi - rotation_deg, 360)
    out = np.empty_like(beam)

    for f in range(beam.shape[0]):
        for t in range(beam.shape[2]):
            out[f, :, t] = np.interp(
                phi_src,
                phi_ext,
                beam_ext[f, :, t],
            )

    return out


# =========================
# 固定北极参考坐标系
# =========================
def build_local_basis(nside, center_pix):
    """
    为某个 HEALPix 像素构造局部坐标基。

    局部 z 轴指向 center_pix。
    固定北极方向作为参考方向。
    若方向过于接近南北极，则叉乘会数值不稳定，因此终止程序。
    """

    z_axis = np.array(hp.pix2vec(nside, center_pix))
    z_axis /= np.linalg.norm(z_axis)

    ref_vec = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(ref_vec, z_axis)) > 0.999:
        print("错误: 太靠近南北极")
        sys.exit(1)

    y_axis = np.cross(z_axis, ref_vec)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    return x_axis, y_axis, z_axis


# =========================
# phi 采样
# =========================
def build_phi_samples(phi_min, phi_range, n):
    """构造 phi 方向的稀疏采样点。"""

    return np.linspace(
        phi_min,
        phi_min + phi_range,
        n,
        endpoint=False,
    )


def find_nearest_phi_indices(phi_grid, phi_samples):
    """为每个 phi 采样点寻找最近的原始波束 phi 网格索引。"""

    return np.array(
        [
            np.argmin(abs(phi_grid - value))
            for value in phi_samples
        ]
    )


# =========================
# 几何预计算
# =========================
def precompute_pixel_geometry(
    nside,
    pix,
    theta_grid,
    phi_grid,
    theta_min,
    theta_max,
    phi_min,
    phi_range,
    n_phi_sample,
):
    """
    为单个中心像素预计算几何映射。

    注意：这里保持原算法不变：
    1. phi 方向用 n_phi_sample 个稀疏采样点；
    2. d_theta 固定为 0.1 deg；
    3. d_phi 固定为 0.1 deg；
    4. patch_solid_angle 使用 n_phi_sample / 3600 * 4*pi。
    """

    x_axis, y_axis, z_axis = build_local_basis(nside, pix)

    theta_idx = np.where(
        (theta_grid >= theta_min)
        & (theta_grid <= theta_max)
    )[0]

    phi_samples = build_phi_samples(
        phi_min,
        phi_range,
        n_phi_sample,
    )

    phi_idx = np.unique(
        find_nearest_phi_indices(
            phi_grid,
            phi_samples,
        )
    )

    theta = np.deg2rad(theta_grid[theta_idx])
    phi = np.deg2rad(phi_grid[phi_idx])

    Phi, Theta = np.meshgrid(
        phi,
        theta,
        indexing="ij",
    )

    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)

    vec = (
        x[..., None] * x_axis
        + y[..., None] * y_axis
        + z[..., None] * z_axis
    ).reshape(-1, 3)

    theta_o = np.arccos(vec[:, 2])
    phi_o = np.mod(
        np.arctan2(vec[:, 1], vec[:, 0]),
        2 * np.pi,
    )

    # 保持原算法不变：这里仍然固定使用 0.1 deg。
    d_theta = np.deg2rad(0.1)
    d_phi = np.deg2rad(0.1)

    dS = (
        np.sin(np.deg2rad(theta_grid[theta_idx]))
        * d_theta
        * d_phi
    )[None, :]

    return {
        "theta_orig": theta_o,
        "phi_orig": phi_o,
        "theta_idx": theta_idx,
        "phi_idx": phi_idx,
        "shape": (phi_idx.size, theta_idx.size),
        "dS": dS,
        "patch_solid_angle": n_phi_sample / 3600 * 4 * np.pi,
    }


# =========================
# 单像素卷积
# =========================
def convolve_one_pixel(geom, hpmap, beam_patch):
    """
    对一个中心像素执行波束卷积。

    hpmap 的单位决定输出单位。
    当前主程序中 hpmap 已经被转换为 mK_CMB。
    """

    sky = hp.get_interp_val(
        hpmap,
        geom["theta_orig"],
        geom["phi_orig"],
    ).reshape(geom["shape"])

    return np.sum(
        beam_patch
        * sky
        * geom["dS"]
    ) / geom["patch_solid_angle"]


# =========================
# 天图生成
# =========================
def build_sky_map_mK_CMB(sky_model, frequency_mhz):
    """
    生成指定频率下的 PySM I 天图，并转换为 mK_CMB。

    若 ADD_CMB_MONOPOLE 为 True，则额外加上 CMB 平均温度。
    """

    sky = sky_model.get_emission(frequency_mhz * u.MHz)[0]

    sky = sky.to(
        u.mK_CMB,
        equivalencies=u.cmb_equivalencies(frequency_mhz * u.MHz),
    ).value

    if ADD_CMB_MONOPOLE:
        sky += CMB_MONOPOLE_MK

    return sky


# =========================
# 单频率处理
# =========================
def process_one_freq(i, fval, beam_cube, geom_cache, sky_model):
    """
    对单个频率执行卷积。

    参数
    ----
    i:
        当前频率索引。

    fval:
        当前频率值，单位 MHz。

    beam_cube:
        波束数组，形状为 (N_freq, N_phi, N_theta)。

    geom_cache:
        所有选中像素的几何缓存。

    sky_model:
        PySM3 天空模型。
    """

    beam = beam_cube[i]

    sky = build_sky_map_mK_CMB(
        sky_model,
        fval,
    )

    res = Parallel(
        n_jobs=N_JOBS,
        prefer="threads",
    )(
        delayed(convolve_one_pixel)(
            g,
            sky,
            beam[np.ix_(g["phi_idx"], g["theta_idx"])],
        )
        for g in geom_cache
    )

    return np.array(res)


# =========================
# 读取波束
# =========================
def read_beam_file(file_path):
    """读取 beam.h5 文件。"""

    with h5py.File(file_path, "r") as f:
        beam_cube = f["data"][:]
        freq = f["frequency"][:]
        beam_phi = f["phi"][:]
        beam_theta = f["theta"][:]

    return beam_cube, freq, beam_phi, beam_theta


# =========================
# 选择目标天区
# =========================
def select_pixels():
    """按照 RA 和 Dec 范围选择需要计算的 HEALPix 像素。"""

    npix = hp.nside2npix(Nside)
    ipix = np.arange(npix)

    theta, phi = hp.pix2ang(Nside, ipix)
    ra = phi
    dec = np.pi / 2 - theta

    mask = (
        (ra > np.deg2rad(Ra_min_deg))
        & (ra < np.deg2rad(Ra_max_deg))
        & (dec > np.deg2rad(Dec_min_deg))
        & (dec < np.deg2rad(Dec_max_deg))
    )

    selected = ipix[mask]

    return selected


# =========================
# 构建几何缓存
# =========================
def build_geometry_cache(selected, beam_theta, beam_phi):
    """为所有选中像素构建几何缓存。"""

    geom_cache = [
        precompute_pixel_geometry(
            Nside,
            pix,
            beam_theta,
            beam_phi,
            theta_start_deg,
            theta_start_deg + theta_max,
            phi_start_deg,
            phi_max_deg,
            N_phi,
        )
        for pix in selected
    ]

    return geom_cache


# =========================
# 还原 HEALPix cube
# =========================
def build_healpix_cube(result, selected, freq):
    """
    将选中像素上的结果还原为完整 HEALPix cube。

    输出形状为 (N_freq, N_pix)。
    未计算区域填入 hp.UNSEEN。
    """

    npix_full = hp.nside2npix(Nside)
    cube = np.full(
        (len(freq), npix_full),
        hp.UNSEEN,
    )

    for i in range(len(freq)):
        cube[i, selected] = result[:, i]

    return cube


# =========================
# 保存结果
# =========================
def save_result(output_path, cube, freq):
    """保存多频 HEALPix map 到 FITS 文件。"""

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


# =========================
# 主程序
# =========================
def main():
    """主流程。"""

    start_time = time.time()

    print("start convolution")

    output_path = build_output_path()

    print(output_path)
    print("N_phi = " + str(N_phi))
    print("ADD_CMB_MONOPOLE = " + str(ADD_CMB_MONOPOLE))

    if ADD_CMB_MONOPOLE:
        print("CMB_MONOPOLE_MK = " + str(CMB_MONOPOLE_MK))

    # 读取波束
    beam_cube, freq, beam_phi, beam_theta = read_beam_file(
        beam_file_path
    )

    # 旋转波束
    beam_cube = rotate_beam_phi_deg(
        beam_cube,
        beam_phi,
        beam_rotation,
    )

    # 天空模型
    sky_model = pysm3.Sky(
        nside=Nside,
        preset_strings=sky_component_list,
    )

    # 选区域
    selected = select_pixels()

    # 几何缓存
    geom_cache = build_geometry_cache(
        selected,
        beam_theta,
        beam_phi,
    )

    # 主循环
    result = np.zeros(
        (len(selected), len(freq)),
    )

    for i, fval in enumerate(freq):
        print(f"{i + 1}/{len(freq)} {fval:.2f} MHz")

        result[:, i] = process_one_freq(
            i,
            fval,
            beam_cube,
            geom_cache,
            sky_model,
        )

    # 还原 HEALPix
    cube = build_healpix_cube(
        result,
        selected,
        freq,
    )

    # 保存
    save_result(
        output_path,
        cube,
        freq,
    )

    print("DONE")
    print("Total time:", (time.time() - start_time) / 60, "min")


if __name__ == "__main__":
    main()