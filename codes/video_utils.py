import glob
import json
import os
import shlex
import subprocess
from enum import Enum
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

import cv2


class imageExtension(Enum):
    """
    Class representing the formats of images supported.
    For more details about PPM, PGM and PBM check this page:
    http://paulbourke.net/dataformats/ppm/
    """
    JPG = 1
    PNG = 2
    PPM = 3
    PGM = 4
    PBM = 5


class videoObj:
    """videoObj class contains important information all important methods and
    tools to access database videos.
    """

    def __init__(self, videopath, annotationFilePath=None):

        self.videopath = videopath
        self.videoInfo = videoInfo(self.videopath)

    def get_frame(self, frame_req, raiseException=True):
        """This method gets the frame of a video and returns a flag informing
        if it was possible, along with the frame itself and the frame size.

        Arguments:
            frame_req {int} -- [Requested frame number to be returned]

        Keyword Arguments:
            raiseException {bool} -- [Flag to raise an exception] (default: {True})

        Returns:
            (ret, frame, frame_size) --
                ret {bool}: whether the frame was read or not,
                frame {np.array}: the frame itself,
                frame_size {list}: frame height, width and #channels
        """

        'Frame count starts from 1 to max frames -> self.infoVideo.getNumberOfFrames()'

        # get the total number of video frames
        nb_frames = self.videoInfo.getNumberOfFrames()

        # sanity check: able to find number of frames?
        if nb_frames is None:
            raise IOError('Unable to find the total number of frames!')

        # sanity check: required frame number is within the video
        if frame_req < 0 or frame_req > int(nb_frames - 1):
            if raiseException is True:
                raise IOError('Required frame={}: Must be between 1 and {}.'.format(
                    frame_req, self.videoInfo.getNumberOfFrames()))
            else:
                print('Error! Required frame={}: must be between 1 and {}.'.format(
                    frame_req, self.videoInfo.getNumberOfFrames()))
                return None, None, None

        # load video
        video_capture = cv2.VideoCapture(self.videopath)

        # get to the frame we want
        # We make frame_req-1, because for this API, our frames go from 1 to max
        # openCV frame count is 0-based
        # Reference:
        # https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
        ret = video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_req)
        if not ret:
            print('fail setting {:d} frame number!'.format(frame_req))

        # read video
        ret, frame = video_capture.read()
        video_capture.release()

        frame_size = None
        if ret:
            frame_size = frame.shape
        else:
            print('fail reading {:d} frame number!'.format(frame_req))
        return ret, frame, frame_size

    def save_frames(self,
                    first_frame,
                    last_frame,
                    frames_skip,
                    output_folder=None,
                    extension=imageExtension.JPG,
                    jpeg_quality=95,
                    compression_level=3,
                    binary_format=True,
                    filename_prefix='frame_'):
        """This method saves the frames between 'first_frame' and 'last_frame'
        (including them) skiping 'frames_skip' frames

        Arguments:
            first_frame {int} -- [Nunber of first frame to save]
            last_frame {int} -- [Number of last frame to save]
            frames_skip {int} -- [Number of frame to skip]


        Keyword Arguments:
            output_folder {str} -- [folder to save frames] (default: {None})
            extension {imageExtension} -- [Object of imageExtension type]
            (default: {imageExtension.JPG})
            jpeg_quality {int} -- [JPEG quality between 0 and 100 the higher the better quality]
            (default: {95})
            compression_level {int} -- [PNG compression level between 0 and 9.
            The higher the value the smaller size and longer compression time.]
            (default: {3})
            binary_format {bool} -- [For PPM, PGM, or PBM, it can be a binary
            format flag] (default:{True})
            filename_prefix {filename prefix} -- [file] (default: {'frame_'})
        """

        # if output folder is not specified, create a folder at the same level
        # where the video is and save the frames in this folder
        # otherwise save where specified
        if output_folder is None:
            output_folder = os.path.splitext(self.videopath)[0] + '_frames'
        else:
            output_folder = os.path.join(
                output_folder,
                os.path.splitext(os.path.basename(self.videopath))[0] + '_frames')

        # create output folder in case it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # check extension and assing params list
        if extension == imageExtension.JPG:
            ext = "jpg"
            ext_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        elif extension == imageExtension.PNG:
            ext = "png"
            ext_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

        elif extension == imageExtension.PPM:
            ext = "ppm"
            ext_params = [cv2.IMWRITE_PXM_BINARY, int(binary_format)]

        elif extension == imageExtension.PGM:
            ext = "pgm"
            ext_params = [cv2.IMWRITE_PXM_BINARY, int(binary_format)]

        elif extension == imageExtension.PBM:
            ext = "pbm"
            ext_params = [cv2.IMWRITE_PXM_BINARY, int(binary_format)]

        # output file name format
        filename_format = '/{prefix}{{:04d}}.{ext}'
        output_path_str = output_folder + filename_format.format(prefix=filename_prefix, ext=ext)

        # loop over frames requested
        for i in range(first_frame, last_frame + 1, frames_skip + 1):

            # Get the ith frame
            res, frame, _ = self.get_frame(i)

            # Check if frame was successfully retrieved and save it
            if res:
                output_path = output_path_str.format(i)
                cv2.imwrite(output_path, frame, ext_params)

                # Save image based on the extension

                if os.path.isfile(output_path):
                    print("File sucessfully saved: %s" % output_path)
                else:
                    print("Error saving file saved: %s" % output_path)

            else:
                print("Error opening the frame %d" % i)

    def cam_params(self,
                   frame_step=0,
                   pattern_size=(9, 6),
                   square_size=1.0,
                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                   debug=False):

        # TODO: Save a file with the cam params
        # so that, we don't need to recalcute these all the time

        print('Calibrating video: ' + self.videoInfo._fileName + ' ...')

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(pattern_size), 3), np.float32)
        objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        objp *= square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        num_pat_found = 0  # number of images where the pattern has been found

        # Frames to scan in order to detect keypoints
        first_frame = 1
        last_frame = self.videoInfo.getNumberOfFrames()

        # loop over frames
        for i in tqdm(range(first_frame, last_frame - 1, frame_step + 1)):

            sleep(0.01)

            # Get the ith frame
            retval, img, _ = self.get_frame(i)

            if not retval:
                tqdm.write('video capture failed!')
                break

            tqdm.write(' Searching for chessboard in frame ' + str(i) + '...')

            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (w, h) = gray.shape

            # cv2.FindChessboardCorners cannot detect chessboard on very large images
            # The likely correct way to proceed is to start at a lower resolution
            # (i.e. downsizing), then scale up the positions of the corners thus found,
            # and use them as the initial estimates for a run of cvFindCornersSubpix at
            # full resolution.
            # (https://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca/15074774)

            # blur before resize
            # blur = cv2.GaussianBlur(gray, (7, 7), 1)

            # resize image
            # TODO: Find a way to compute the best way to compute scale_factor
            # maybe put the image in a standard size before finding keypoints
            scale_factor = .3
            gray_small = cv2.resize(
                gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Find the chess board corners
            ret, corners_small = cv2.findChessboardCorners(
                gray_small,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret is True:

                tqdm.write('pattern found')

                # scale up the positions
                corners = corners_small / scale_factor

                # update number of images where the pattern has been found
                num_pat_found += 1

                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints.append(corners)

                if debug is True:

                    # print('Searching for chessboard in frame ' + str(i) + '...')
                    # Draw and display the corners
                    # img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # plt.axis('off')
                    # plt.show()

                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.scatter(x=corners[:, :, 0], y=corners[:, :, 1], c='r', s=20)
                    plt.show()
                    tqdm.write(str(img.shape))
            else:
                tqdm.write('pattern NOT found')

                if debug is True:
                    # Draw and display the corners
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                    tqdm.write(str(img.shape))

        print('Number of pattern found: ', num_pat_found)

        # Camera calibration
        print('computing cam params...')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        print('Done!')

        return ret, mtx, dist, rvecs, tvecs

    def calibration(self, cam_params=cam_params):

        video_dir = os.path.dirname(self.videoInfo.getFilePath())
        calibrated_vid_path = glob.glob(os.path.join(video_dir, 'undistort*'))

        # check if the undistorted video already exists
        if calibrated_vid_path:
            print(calibrated_vid_path)
            return videoObj(calibrated_vid_path[0])
        else:
            pass


class videoInfo(object):
    """
    videoInfo brings all important information about a video from videos database.

    strongly based on:
    https://github.com/rafaelpadilla/DeepLearning-VDAO/blob/master/VDAO_Access/VDAOHelper.py
    """

    def __init__(self, video_file):
        # class constructor
        self._filePath = video_file
        # Inicializa variáveis
        self._idxVideoInfo = None
        self._idxAudioInfo = None
        self._idxSubtitleInfo = None
        self._fileName = None
        self._format = None
        self._formatLong = None
        self._size = None  # in bytes
        self._codec = None
        self._codecLong = None
        self._width = None
        self._height = None
        self._widthHeight = None
        self._sampleAspectRatio = None
        self._displayAspectRatio = None
        self._pixelFormat = None
        self._frameRate = None
        self._framesPerSecond = None
        self._durationTS = None
        self._duration = None
        self._durationReal = None
        self._bitRate = None
        self._numberOfFrames = None
        self._createdOn = None
        self._enconder = None

        try:
            with open(os.devnull, 'w') as tempf:
                subprocess.check_call(["ffprobe", "-h"], stdout=tempf, stderr=tempf)
        except IOError:
            raise IOError('ffprobe not found!')

        if os.path.isfile(video_file):

            cmd = "ffprobe -v error -print_format json -show_streams -show_format"

            # makes a list with contents of cmd
            args = shlex.split(cmd)
            # append de video file to the list
            args.append(video_file)

            # Running ffprobe process and loads it in a json structure
            ffoutput = subprocess.check_output(args).decode('utf-8')
            ffoutput = json.loads(ffoutput)

            # Check available information on the file
            for i in range(len(ffoutput['streams'])):
                if ffoutput['streams'][i]['codec_type'] == 'video':
                    self._idxVideoInfo = i
                elif ffoutput['streams'][i]['codec_type'] == 'audio':
                    self._idxAudioInfo = i
                elif ffoutput['streams'][i]['codec_type'] == 'subtitle':
                    self._idxSubtitleInfo = i

            # Set properties related to the file itself
            self._fileName = ffoutput['format']['filename']
            self._fileName = self._fileName[self._fileName.rfind('/') + 1:]
            self._format = ffoutput['format']['format_name']
            self._formatLong = ffoutput['format']['format_long_name']
            self._size = ffoutput['format']['size']
            if 'creation_time' in ffoutput['format']['tags']:
                self._createdOn = ffoutput['format']['tags']['creation_time']
            if 'encoder' in ffoutput['format']['tags']:
                self._encoder = ffoutput['format']['tags']['encoder']

            # Set properties related to the video
            if self.isVideo():
                self._codec = ffoutput['streams'][self._idxVideoInfo]['codec_name']
                self._codecLong = ffoutput['streams'][self._idxVideoInfo]['codec_long_name']
                self._width = ffoutput['streams'][self._idxVideoInfo]['width']
                self._height = ffoutput['streams'][self._idxVideoInfo]['height']
                self._widthHeight = [self._width, self._height]
                # self._sampleAspectRatio = ffoutput['streams'][self._idxVideoInfo][
                # 'sample_aspect_ratio']
                # self._displayAspectRatio = ffoutput['streams'][self._idxVideoInfo][
                # 'display_aspect_ratio']
                self._pixelFormat = ffoutput['streams'][self._idxVideoInfo]['pix_fmt']
                self._frameRate = ffoutput['streams'][self._idxVideoInfo]['r_frame_rate']
                self._framesPerSecond = int(self._frameRate[:self._frameRate.index('/')])
                self._durationTS = ffoutput['streams'][self._idxVideoInfo]['duration_ts']
                self._duration = ffoutput['streams'][self._idxVideoInfo]['duration']
                self._bitRate = ffoutput['streams'][self._idxVideoInfo]['bit_rate']
                self._numberOfFrames = ffoutput['streams'][self._idxVideoInfo]['nb_frames']
        else:
            raise IOError('This is not a valid media file ' + video_file)

    def isVideo(self):
        """Returns true if the file is a valid video extension"""
        val = False
        if self._idxVideoInfo is not None:
            val = True
        return val

    def hasAudio(self):
        """Returns true if the file provides audio information"""
        val = False
        if self._idxAudioInfo is not None:
            val = True
        return val

    def hasSubtitles(self):
        """Returns true if the file makes subtitle data available"""
        val = False
        if self._idxSubtitleInfo is not None:
            val = True
        return val

    def getFilePath(self):
        """Gets full file path"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._filePath
        return val

    def getFileName(self):
        """Gets the name of the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._fileName
        return val

    def getFormat(self):
        """Gets format of the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._format
        return val

    def getFormatLong(self):
        """Gets full format description"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._formatLong
        return val

    def getSize(self):
        """Gets the size of the file in bytes"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._size
        return val

    def getCreationDate(self):
        """Gets the creation date and time"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._createdOn
        return val

    def getEnconderType(self):
        """Gets the encoder used to generate the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._encoder
        return val

    def getCodecType(self):
        """Gets the codec for the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._codec
        return val

    def getCodecLongType(self):
        """Gets the full description of the codec"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._codecLong
        return val

    def getWidth(self):
        """Gets the width (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._width
        return val

    def getHeight(self):
        """Gets the height (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._height
        return val

    def getWidthHeight(self):
        """Gets the width and height (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._widthHeight
        return val

    def getSampleAspectRatio(self):
        """Gets width by height ratio of the pixels with respect to the original source"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._sampleAspectRatio
        return val

    def getDisplayAspectRatio(self):
        """Gets width by height ratio of the data as it is supposed to be displayed"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._displayAspectRatio
        return val

    def getPixelFormat(self):
        """Gets the raw representation of the pixel.
           For reference see: http://blog.arrozcru.org/?p=234"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._pixelFormat
        return val

    def getFrameRateFloat(self):
        """Gets number of frames that are displayed per second in float format"""
        val = self.getFrameRate()
        if val is not None:
            idx = val.find('/')
            if idx == -1:
                return None
            num = float(val[:idx])
            den = float(val[idx + 1:])
            return num / den
        return val

    def getFrameRate(self):
        """Gets number of frames that are displayed per second in the format X/1"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._frameRate
        return val

    # def getFramesPerSecond(self): #WRONG!
    #     return None # Make it useless
    #     """Gets number of frames that are displayed per second ????? TO REVIEW!"""
    #     val = None
    #     if self._idxVideoInfo is not None:
    #         val = self._framesPerSecond
    #     return val

    def getDurationTs(self):
        """Gets the duration whole video in frames ?????"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._durationTS
        return val

    def getRealDuration(self):
        """Gets the full duration of the video in seconds"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._durationReal
        return val

    def getBitRate(self):
        """Gets the number of bits used to represent each second of the video"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._bitRate
        return val

    def getNumberOfFrames(self):
        """Gets the number of frames of the whole video ????"""
        val = None
        if self._idxVideoInfo is not None:
            val = int(self._numberOfFrames)
        return val

    def printAllInformation(self):

        print('\n*************************************')
        print('************* video info ************')
        print(' ')
        print('File path: ' + str(self._filePath))
        print('File name: ' + str(self._fileName))
        print('File extension: ' + str(self._format) + ' (' + str(self._formatLong) + ')')
        print('Created on: ' + str(self._createdOn))
        # print('Encoder: ' + str(self._encoder))
        print('File size: ' + str(self._size))
        print('Codec: ' + str(self._codec) + ' (' + str(self._codecLong) + ')')
        print('Width: ' + str(self._width))
        print('Height: ' + str(self._height))
        print('Width x Height: ' + str(self._widthHeight))
        print('Sample aspect ratio: ' + str(self._sampleAspectRatio))
        print('Display aspect ratio: ' + str(self._displayAspectRatio))
        print('Pixel format: ' + str(self._pixelFormat))
        print('Frame rate: ' + str(self._frameRate))
        print('Duration ts: ' + str(self._durationTS))
        print('Duration: ' + str(self._duration))
        # print('Real douration: ' + str(self._real))
        print('Bit rate: ' + str(self._bitRate))
        print('Number of frames: ' + str(self._numberOfFrames))

        print('\n*************************************')
