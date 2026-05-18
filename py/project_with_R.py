import numpy as np

def project_with_R(noise, R, los_axis=-1):
    """
    将 PCA 投影矩阵 R 应用于 noise（严格对齐 pca_clean）

    参数:
        noise: ndarray, shape = (3D)
        R: ndarray, shape = (N_freq, N_freq)
        los_axis: int, 频率轴位置

    返回:
        noise_clean: ndarray, same shape as noise
    """

    assert noise.ndim == 3, "noise must be 3D"

    if los_axis < 0:
        los_axis = 3 + los_axis

    # ===== 1. 构造与 pca_clean 一致的轴顺序 =====
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = [los_axis] + axes

    # ===== 2. 转置：把频率轴放到第 0 维 =====
    noise_t = np.transpose(noise, axes=axes)

    nz, nx, ny = noise_t.shape

    # ===== 3. reshape → (N_freq, N_pix) =====
    noise_2d = noise_t.reshape(nz, -1)

    # ===== 4. PCA 投影 =====
    noise_clean_2d = R @ noise_2d

    # ===== 5. reshape 回 3D =====
    noise_clean_t = noise_clean_2d.reshape(nz, nx, ny)

    # ===== 6. 转回原始轴顺序 =====
    noise_clean = np.transpose(noise_clean_t, axes=np.argsort(axes))

    return noise_clean



# import numpy as np

# def project_with_R(noise, R, los_axis=-1):
#     """Apply PCA projection matrix R to noise (strictly aligned with pca_clean)"""
#     assert noise.ndim == 3, "noise must be 3D"  # ensure input is 3D
#     if los_axis < 0:
#         los_axis = 3 + los_axis  # convert negative axis index to positive

#     axes = [0, 1, 2]; axes.remove(los_axis); axes = [los_axis] + axes  # reorder axes so freq axis is first
#     noise_t = np.transpose(noise, axes=axes)  # move frequency axis to dim 0

#     nz, nx, ny = noise_t.shape  # extract dimensions
#     noise_2d = noise_t.reshape(nz, -1)  # reshape to (N_freq, N_pix)

#     noise_clean_2d = R @ noise_2d  # apply PCA projection
#     noise_clean_t = noise_clean_2d.reshape(nz, nx, ny)  # reshape back to 3D

#     noise_clean = np.transpose(noise_clean_t, axes=np.argsort(axes))  # restore original axis order
#     return noise_clean  # return cleaned noise