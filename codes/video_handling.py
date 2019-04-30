import glob
import json
import os
import shlex
import subprocess
import time
import warnings
from enum import Enum

import cv2
import numpy as np
from tqdm.autonotebook import tqdm

from annotation import Annotation
from utils.calibration import compute_cam_params, find_chessboard_kpts_video
from utils.img_utils import add_bb_on_image

warnings.filterwarnings("ignore")


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

    def __init__(self, videopath, annotation_path=None):

        self.videopath = videopath
        self.videoInfo = videoInfo(self.videopath)

        self._annotation = Annotation(
            annotation_path=annotation_path, total_frames=self.videoInfo.getNumberOfFrames())

    def parse_annotation(self):
        return self._annotation._parse_file()

    def get_annotations(self):
        if self._annotation.parsed is False:
            self.parse_annotation()
        return self._annotation

    def get_frame_annotations(self, frame_idx):
        annotation = self.get_annotations()
        annotation = annotation.annotation_dict
        return annotation['frame_{:04d}'.format(frame_idx)]

    def get_batch_annotations(self, batch_size=8):
        annotation = self.get_annotations()
        annotation = annotation.annotation_dict

        batch_annotations = {}
        # frame = 0

        for k, v in annotation.items():

            batch_annotations[k] = v

            if len(batch_annotations) >= batch_size:
                yield batch_annotations
                batch_annotations = {}

        # while frame < self.videoInfo.getNumberOfFrames():
        #     batch_annotations = {}
        #     for i in range(batch_size):

        #         try:
        #             batch_annotations['frame_{:04d}'.format(frame)] = annotation
        #             ['frame_{:04d}'.format(frame)]

        #         except KeyError:
        #             print('Not found annotation for frame {:04d}'.format(frame))
        #             break

        #         frame += 1

        #     yield batch_annotations

    def set_annotation(self, new_annotation):
        self._annotation = new_annotation
        self._annotation.parsed = True
        self._annotation.error = False

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
        # openCV frame count is 0-based: our frames go from 0 to max - 1
        # Reference:
        # https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html
        ret = video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_req)
        if not ret:
            print('fail setting {:d} frame number!'.format(frame_req))

        # read video
        ret, frame = video_capture.read()
        # self._video_capture.release()

        frame_size = None
        if ret:
            frame_size = frame.shape
        else:
            print('fail reading {:d} frame number!'.format(frame_req))
        return ret, frame, frame_size

    def get_all_frames(self):
        """Returns a np array of size NxHxWxC
        N- #frames
        H- Height
        W- Width
        C- #Channels (BGR order)

        Returns:
            Tuple -- Returns a numpy array of size NxHxWxC
        """

        frames = []

        video_capture = cv2.VideoCapture(self.videopath)

        ret, frame = video_capture.read()

        while ret is True:

            frames.append(frame)
            ret, frame = video_capture.read()

        frames = np.stack(np.array(frames), axis=0)

        # frames = np.stack([np.array(img) for (_, img) in video_capture.read()], axis=0)

        return frames

    def get_batch_frames(self, batch_size=8):
        """Returns a generator (np array of size NxHxWxC)
        N- batch size (number of frames)
        H- Height
        W- Width
        C- #Channels (BGR order)

        Returns:
            Tuple -- Returns a generator (np array of size NxHxWxC)
        """

        video_capture = cv2.VideoCapture(self.videopath)
        nb_frames = self.videoInfo.getNumberOfFrames()

        # check the number of batches
        n_batches = int(np.ceil(nb_frames / batch_size))
        print("{} batches of size {}".format(n_batches, batch_size))

        # print a warning if last batch has different size
        if (nb_frames % batch_size) != 0:
            print('Warning: Last batch has different size')

        ret = True

        while ret is True:

            frames = []

            for i in range(batch_size):
                ret, frame = video_capture.read()
                if not ret:
                    break
                frames.append(frame)

            batch = np.stack(np.array(frames), axis=0)

            yield batch

        # frames = np.stack([np.array(img) for (_, img) in video_capture.read()], axis=0)

    def play_video(self, show_bb=False):

        annot = self.get_annotations()

        if show_bb and annot.parsed is False:
            # if somehow there was an error while parsing, do not show bounding boxes
            show_bb = self.parse_annotation()
        print(self.videopath)
        video_capture = cv2.VideoCapture(self.videopath)

        fps = self.videoInfo.getFrameRateFloat()  # or cap.get(cv2.CAP_PROP_FPS)

        wait_fraction = int(
            770 /
            fps)  # adjust the 770 factor in order to try the display the video at original fps

        ret, frame = video_capture.read()

        frame_idx = 0

        while ret is True:

            start_time = time.time()
            key = cv2.waitKey(1) & 0xFF

            if show_bb:

                frame_annot = annot.get_annoted_frame(frame_idx)

                for object_name, bb in frame_annot.items():
                    frame = add_bb_on_image(frame, bb, label=object_name)

            delta_time = (time.time() - start_time) * 1000  # secs to ms
            wait_ms = wait_fraction - delta_time

            # pause
            if key == ord('p'):
                while True:
                    key2 = cv2.waitKey(1) or 0xFF
                    cv2.imshow(self.videopath, frame)

                    if key2 == ord('p'):
                        break

            # Show frame
            cv2.imshow(self.videopath, frame)
            cv2.waitKey(int(wait_ms))  # in miliseconds
            print(frame_idx)

            # read next frame
            ret, frame = video_capture.read()
            frame_idx += 1

            if key == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def play_frame_by_frame(self, show_bb=False):

        print('Playing video frame by frame \n')
        print('press \'s\' to next frame')
        print('press \'a\' to previos frame')
        print('press \'q\' to quit\n')

        annot = self.get_annotations()

        if show_bb and annot.parsed is False:
            # if somehow there was an error while parsing, do not show bounding boxes
            show_bb = self.parse_annotation()

        # get the total number of video frames
        nb_frames = self.videoInfo.getNumberOfFrames()

        frame_idx = 0

        while frame_idx < int(nb_frames):

            res, frame, _ = self.get_frame(frame_idx)

            if show_bb:

                frame_annot = annot.get_annoted_frame(frame_idx)

                for object_name, bb in frame_annot.items():
                    frame = add_bb_on_image(frame, bb, label=object_name)

            cv2.imshow('Frame{:04d}'.format(frame_idx), frame)

            wkey = cv2.waitKey(0)
            key = chr(wkey % 256)

            if key == 'a':
                frame_idx -= 1
            elif key == 's':
                frame_idx += 1
            elif key == 'q':
                cv2.destroyAllWindows()
                return

    def save_frames(self,
                    first_frame=0,
                    last_frame=None,
                    every=1,
                    output_folder=None,
                    extension=imageExtension.JPG,
                    jpeg_quality=95,
                    compression_level=3,
                    binary_format=True,
                    filename_prefix='frame_',
                    verbose=False):
        """This method saves the frames between 'first_frame' and 'last_frame'
        (including them) at 'every' frames

        Arguments:
            first_frame {int} -- [Nunber of first frame to save]
            last_frame {int} -- [Number of last frame to save]
            every {int} -- [save at 'every' number of frames]


        Keyword Arguments:
            output_folder {str} -- [folder to save frames] (default: {None})
            extension {imageExtension} -- [Object of imageExtension type]
            (default: {imageExtension.JPG})
            jpeg_quality {int} -- [JPEG quality between 0 and 100 the higher the better quality]
            (default: {95})
            compression_level {int} -- [PNG compression level between 0 and 9.
            The higher the value the smaller size ancam_pa longer compression time.]
            (default: {3})
            binary_format {bool} -- [For PPM, PGM, ocam_pa PBM, it can be a binary
            format flag] (default:{True})
            filename_prefix {filename prefix} -- [ficam_pae] (default: {'frame_'})
        """

        # if output folder is not specified, create cam_pa folder at the same level
        # where the video is and save the frames in cam_pahis folder
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

        if last_frame is None:
            last_frame = self.videoInfo.getNumberOfFrames()

        # loop over frames requested
        for i in tqdm(range(first_frame, last_frame, every)):

            # Get the ith frame
            res, frame, _ = self.get_frame(i)

            # Check if frame was successfully retrieved and save it
            if res:
                output_path = output_path_str.format(i)
                cv2.imwrite(output_path, frame, ext_params)

                # Save image based on the extension

                if os.path.isfile(output_path) and verbose:
                    tqdm.write("File sucessfully saved: %s" % output_path)
                else:
                    tqdm.write("Error saving file saved: %s" % output_path)

            else:
                tqdm.write("Error opening the frame %d" % i)

    def cam_params(self,
                   first_frame=0,
                   last_frame=None,
                   every=1,
                   pattern_size=(9, 6),
                   square_size=1.0,
                   alpha=0,
                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001),
                   debug=False,
                   verbose=False):

        objpoints, imgpoints, img_size = objpoints, imgpoints, img_size = find_chessboard_kpts_video(
            self.videopath,
            first_frame=first_frame,
            last_frame=last_frame,
            every=every,
            pattern_size=pattern_size,
            square_size=square_size,
            criteria=criteria,
            debug=debug,
            verbose=verbose)

        cam_params = compute_cam_params(objpoints, imgpoints, img_size, alpha=alpha)

        return cam_params

    def calibration(self, cam_params=cam_params):

        video_dir = os.path.dirname(self.videoInfo.getFilePath())
        calibrated_vid_path = glob.glob(os.path.join(video_dir, 'rect*'))

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
        # print('Real duration: ' + str(self._real))
        print('Bit rate: ' + str(self._bitRate))
        print('Number of frames: ' + str(self._numberOfFrames))

        print('\n*************************************')
