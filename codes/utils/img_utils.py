from skimage.measure import compare_mse, compare_psnr, compare_ssim

import cv2


def add_bb_on_image(image, bounding_box, color=(255, 0, 0), thicknes=3, label=None):

    # color
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    # font params
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.5
    # font_thickness = 1

    # bb limits
    x_i = bounding_box[0]
    y_i = bounding_box[1]
    x_o = bounding_box[2]
    y_o = bounding_box[3]

    # draw bounding box
    cv2.rectangle(image, (x_i, y_i), (x_o, y_o), (b, g, r), thicknes)

    # TODO: put text
    if label is not None:
        pass
    return image


def compute_mse(img1, img2):
    return compare_mse(img1, img2)


def compute_psnr(img1, img2):
    return compare_psnr(img1, img2)


def compute_ssim(img1, img2, multichannel=True):
    return compare_ssim(img1, img2, multichannel=multichannel)


def rectify_img(img, cam_params):

    dst = cv2.undistort(img, cam_params['mtx'], cam_params['dist'], None)

    return dst