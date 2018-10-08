from skimage.measure import compare_mse, compare_psnr, compare_ssim


def compute_mse(frame1, frame2):
    return compare_mse(frame1, frame2)


def compute_psnr(frame1, frame2):
    return compare_psnr(frame1, frame2)


def compute_ssim(frame1, frame2, multichannel=True):
    return compare_ssim(frame1, frame2, multichannel=multichannel)
