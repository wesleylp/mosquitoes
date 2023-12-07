import os
import xml.etree.ElementTree as ET
from .annotation import Annotation, AnnotationImage


class CVATAnnotation(Annotation):
    def __init__(
        self,
        annotation_path=None,
        total_frames=None,
        width_height=None,
        encoding="utf8",
    ):
        self.total_frames = total_frames
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False
        self.width_height = width_height

        self.__all_objects = dict()

    def __open_xml_file(self, filename):
        tree = ET.parse(filename)
        return tree.getroot()

    def __get_object_name(self, track):
        return track.attrib["label"]

    def __update_count_objs(self, obj_name):
        if obj_name not in self.__all_objects.keys():
            self.__all_objects[obj_name] = 0

        self.__all_objects[obj_name] += 1

    def _parse_file(self):
        if (self.annotation_path is None) or (
            os.path.exists(self.annotation_path) is False
        ):
            return False

        # create dictonary with number of frames
        self.annotation_dict = {
            "frame_{:05d}".format(d): {} for d in range(self.total_frames)
        }
        self.objects = dict()

        # reading annotation file
        annotation_file = self.__open_xml_file(self.annotation_path)

        tracks = annotation_file.findall("track")

        for track in tracks:
            object_name = self.__get_object_name(track)
            self.__update_count_objs(object_name)

            object_name = f"{object_name}-{self.__all_objects[object_name]:03d}"

            if object_name not in self.objects.keys():
                self.objects[object_name] = dict()
                self.objects[object_name]["boxes"] = []
                self.objects[object_name]["frames"] = []

            boxes = track.findall("box")

            # attributes are considered immutable in the video
            # if that is not the case, you should implement a way to update them
            attributes = boxes[0].findall("attribute")
            self.objects[object_name]["attributes"] = {
                attrib.attrib["name"]: attrib.text for attrib in attributes
            }

            for box in boxes:
                info = box.attrib

                try:
                    outside = int(info["outside"])
                except KeyError:
                    outside = False

                # occluded = int(info['occluded'])

                if outside:
                    continue

                frame_idx = int(info["frame"])
                xtl = int(float(info["xtl"]))
                xbr = int(float(info["xbr"]))
                ytl = int(float(info["ytl"]))
                ybr = int(float(info["ybr"]))

                bb = [xtl, ytl, xbr, ybr]

                self.annotation_dict["frame_{:05d}".format(frame_idx)][object_name] = bb
                self.objects[object_name]["boxes"].append(bb)
                self.objects[object_name]["frames"].append(frame_idx)

        self.parsed = True
        self.error = False

        return True


class CVATAnnotationImage(AnnotationImage):
    def __init__(self, frame_number, annotation_path=None, encoding="utf8"):
        # self.total_frames = total_frames
        self.frame_number = frame_number
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False

        self.__all_objects = dict()

    def __open_xml_file(self, filename):
        tree = ET.parse(filename)
        return tree.getroot()

    def __get_object_name(self, track):
        return track.attrib["label"]

    def __update_count_objs(self, obj_name):
        if obj_name not in self.__all_objects.keys():
            self.__all_objects[obj_name] = 0

        self.__all_objects[obj_name] += 1

    def _parse_file(self):
        if (self.annotation_path is None) or (
            os.path.exists(self.annotation_path) is False
        ):
            return False

        # # create dictonary with number of frames
        # self.annotation_dict = {'frame_{:05d}'.format(d): {} for d in range(self.total_frames)}
        self.objects = dict()

        # reading annotation file
        annotation_file = self.__open_xml_file(self.annotation_path)

        tracks = annotation_file.findall("track")

        for track in tracks:
            object_name = self.__get_object_name(track)
            self.__update_count_objs(object_name)

            object_name = f"{object_name}-{self.__all_objects[object_name]:03d}"

            if object_name not in self.objects.keys():
                self.objects[object_name] = dict()
                self.objects[object_name]["boxes"] = []
                self.objects[object_name]["frames"] = []

            boxes = track.findall("box")

            attributes = boxes[0].findall("attribute")
            self.objects[object_name]["attributes"] = {
                attrib.attrib["name"]: attrib.text for attrib in attributes
            }

            for box in boxes:
                info = box.attrib

                outside = int(info["outside"])
                occluded = int(info["occluded"])

                if outside:
                    continue

                frame_idx = int(info["frame"])

                if frame_idx == self.frame_number:
                    xtl = int(float(info["xtl"]))
                    xbr = int(float(info["xbr"]))
                    ytl = int(float(info["ytl"]))
                    ybr = int(float(info["ybr"]))

                    bb = [xtl, ytl, xbr, ybr]

                    self.annotation_dict["{}".format(object_name)] = bb

        self.parsed = True
        self.error = False

        return True


if __name__ == "__main__":
    annot = CVATAnnotation(
        "/home/wesley.passos/repos/mosquitoes-wes/data/fiverr/annotations_sampled/20190601_rectified_DJI_0003_sampled.xml",
        total_frames=328,
        width_height=(4096, 2160),
    )

    annot._parse_file()

    annot_image = CVATAnnotationImage(
        1740,
        "/home/wesley.passos/repos/mosquitoes-wes/data/v1.0/annotation/deleteme.xml",
    )
    boxes, labels = annot_image.get_bboxes_labels()

    print(annot)
