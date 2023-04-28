import xml.etree.ElementTree as ET
import os
import imageio
import pickle
from video_handling import videoObj
import copy
from utils.time_consist import video_phaseCorrelation, get_shift
import numpy as np
import argparse
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Time Consist")

    # TODO: make in batch ()
    parser.add_argument(
        "--video_path",
        default=
        '/home/wesley.passos/repos/mosquitoes-wes/data/fiverr/videos/20210203_rectfied_DJI_0059.avi',
        metavar="FILE",
        help="path to video")

    args = parser.parse_args()

    video_path = args.video_path
    video_name = os.path.basename(video_path)

    # TODO: include this in the argparse
    annotation_filepath = f'/home/wesley.passos/repos/mosquitoes-wes/data/fiverr/annotations/{video_name.replace(".avi", "_sampled.xml")}'

    # get video information
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    # nb_frames = int(reader.get_meta_data()['duration'] * fps)

    #### PHASE CORRELATION ####
    vid = videoObj(video_path)
    video_width = vid.videoInfo.getWidth()
    video_height = vid.videoInfo.getHeight()
    nb_frames = int(vid.videoInfo.getNumberOfFrames())

    print('Phase correlation...')
    phase_corr_folder = os.path.dirname(video_path).replace('videos', 'phase_correlation')
    os.makedirs(phase_corr_folder, exist_ok=True)

    phase_corr_file = os.path.join(os.path.join(phase_corr_folder),
                                   video_name.replace(".avi", "_phaseCorr.pkl"))

    # in case it was previously computed, load it
    if os.path.isfile(phase_corr_file):
        print('loading phase corr...')
        pkl_file = open(phase_corr_file, 'rb')
        phase_corr = pickle.load(pkl_file)
    # otherwise, compute and save for future usage
    else:
        print('computing phase corr...')
        pkl_file = open(phase_corr_file, 'wb')
        phase_corr = video_phaseCorrelation(vid, scale=0.4)
        pickle.dump(phase_corr, pkl_file)

    ############

    # load annotations
    tree = ET.parse(annotation_filepath)
    root = tree.getroot()

    # set the last frame as the total number of (downsampled) video frames
    stop_frame = int(tree.find('.//stop_frame').text)
    tree.find('.//stop_frame').text = str(nb_frames - 1)

    tracks = root.findall('track')
    # iterate over tracks
    print('Upsampling annotations...')
    for track in tqdm(tracks):
        boxes = track.findall('box')

        # interpolate backwards the first object appearance
        box_to_backpropagate = copy.deepcopy(boxes[0].attrib)
        frame = int(box_to_backpropagate['frame'])
        if frame > 0:
            new_frame = (frame * 30) - 1
            while new_frame >= 0 and new_frame % 30 != 0:
                x_shift, y_shift = get_shift(phase_corr, new_frame + 1, -1)

                # add the shift in the box coordinates
                xtl = float(box_to_backpropagate['xtl']) + x_shift
                xbr = float(box_to_backpropagate['xbr']) + x_shift
                ytl = float(box_to_backpropagate['ytl']) + y_shift
                ybr = float(box_to_backpropagate['ybr']) + y_shift

                # clip the box to get inside the frame
                xtl = np.clip(xtl, 0, video_width)
                xbr = np.clip(xbr, 0, video_width)
                ytl = np.clip(ytl, 0, video_height)
                ybr = np.clip(ybr, 0, video_height)

                object_width = xbr - xtl
                object_height = ybr - ytl

                if object_width < 20 or object_height < 20:
                    break

                box_to_backpropagate['frame'] = str(f'{int(new_frame)}')

                box_to_backpropagate['xtl'] = str(f'{xtl:.2f}')
                box_to_backpropagate['xbr'] = str(f'{xbr:.2f}')
                box_to_backpropagate['ytl'] = str(f'{ytl:.2f}')
                box_to_backpropagate['ybr'] = str(f'{ybr:.2f}')

                upsampled_box = ET.SubElement(track, 'box', box_to_backpropagate)
                upsampled_box.text = boxes[0].text
                upsampled_box.tail = boxes[0].tail

                new_frame -= 1
                box_to_backpropagate = copy.deepcopy(box_to_backpropagate)

        # propagate forward

        # iterate over bboxes
        for box_idx, box_labeled in enumerate(boxes):
            # First, set the manually labeled frame back to the original video
            frame = int(box_labeled.attrib['frame'])
            box_labeled.set("frame", f"{int(frame*30)}")

            if box_labeled.attrib["outside"] == "1":
                track.remove(box_labeled)

            elif box_labeled.attrib["outside"] == "0":
                new_frame = (frame * 30) + 1
                box_propagated_attrib = copy.deepcopy(box_labeled.attrib)

                while new_frame % 30 != 0 and new_frame < nb_frames:
                    x_shift, y_shift = get_shift(phase_corr, new_frame - 1, 1)

                    # add the shift in the box coordinates
                    xtl = float(box_propagated_attrib['xtl']) + x_shift
                    xbr = float(box_propagated_attrib['xbr']) + x_shift
                    ytl = float(box_propagated_attrib['ytl']) + y_shift
                    ybr = float(box_propagated_attrib['ybr']) + y_shift

                    # clip the box to get inside the frame
                    xtl = np.clip(xtl, 0, video_width)
                    xbr = np.clip(xbr, 0, video_width)
                    ytl = np.clip(ytl, 0, video_height)
                    ybr = np.clip(ybr, 0, video_height)

                    object_width = xbr - xtl
                    object_height = ybr - ytl

                    box_propagated_attrib['frame'] = str(f'{int(new_frame)}')

                    # drone warming up (rotating)
                    if new_frame < 30 or box_idx < 10:
                        box_propagated_attrib['keyframe'] = "0"
                    else:
                        box_propagated_attrib['keyframe'] = "1"

                    box_propagated_attrib['xtl'] = str(f'{xtl:.2f}')
                    box_propagated_attrib['xbr'] = str(f'{xbr:.2f}')
                    box_propagated_attrib['ytl'] = str(f'{ytl:.2f}')
                    box_propagated_attrib['ybr'] = str(f'{ybr:.2f}')

                    box_propagated = ET.SubElement(track, 'box', box_propagated_attrib)
                    box_propagated.text = box_labeled.text
                    box_propagated.tail = box_labeled.tail

                    if object_width < 20 or object_height < 20:
                        box_propagated_attrib['keyframe'] = "1"
                        box_propagated_attrib['outside'] = "1"
                        break

                    new_frame += 1
                    box_propagated_attrib = copy.deepcopy(box_propagated_attrib)

        # sort by frame
        sorted_track = sorted(track, key=lambda child: int(child.attrib['frame']))
        track[:] = sorted_track

        # make all boxes "appear"
        boxes = track.findall('box')
        [b.set("outside", "0") for b in boxes]

        boxes[-1].set("keyframe", "1")
        boxes[0].set("keyframe", "1")

        # set the last one as 1, except if it is the last frame
        if boxes[-1].attrib["frame"] != str(nb_frames - 1):
            boxes[-1].set("outside", "1")

    # save file
    tree.write(video_path.replace('.avi', '.xml'),
               encoding="utf-8",
               xml_declaration=True,
               short_empty_elements=False)
    print('end')
