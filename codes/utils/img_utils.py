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


def find_chessboard_keypoints_img(img,
                                  pattern_size=(9, 6),
                                  square_size=1.0,
                                  criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30,
                                            0.001),
                                  debug=False,
                                  verbose=False):

    # get image size
    h, w = img.shape[:2]

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.FindChessboardCorners cannot detect chessboard on very large images
    # The likely correct way to proceed is to start at a lower resolution
    # (i.e. downsizing), then scale up the positions of the corners thus found,
    # and use them as the initial estimates for a run of cvFindCornersSubpix at
    # full resolution.
    # (https://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca/15074774)

    # resize image
    # TODO: Find a way to compute the best way to compute scale_factor
    # maybe put the image in a standard size before finding keypoints
    scale_factor = .3
    gray_small = cv2.resize(
        gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    corners = None

    # Find the chess board corners
    ret, corners_small = cv2.findChessboardCorners(
        gray_small,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if ret is True:

        # scale up the positions
        corners = corners_small / scale_factor

        corners = cv2.cornerSubPix(gray, corners, (23, 23), (-1, -1), criteria)

    return ret, corners, w, h


def rectify_img(img, cam_params):

    # undistorting image
    # we use only k1,k2, p1, and p2 dist coeff because using more coefs can lead to numerical instability
    dst = cv2.undistort(img, cam_params['mtx'], cam_params['dist'][0][:4], None,
                        cam_params['newcameramtx'])

    # crop and save the image
    x, y, w, h = cam_params['roi']
    dst = dst[y:y + h, x:x + w]

    return dst
