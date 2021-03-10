import os
from boxes import clip_box_to_image


class Annotation:
    def __init__(self,
                 annotation_path=None,
                 total_frames=None,
                 width_height=None,
                 encoding='ISO-8859-1'):
        self.total_frames = total_frames
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False
        self.width_height = width_height

    def _parse_file(self):
        if (self.annotation_path is None) or (os.path.exists(self.annotation_path) is False):
            return False

        # create dictonary with number of frames
        self.annotation_dict = {'frame_{:05d}'.format(d): {} for d in range(self.total_frames)}
        self.objects = dict()

        # reading annotation file
        with open(self.annotation_path, encoding=self.encoding) as annotation_file:

            # reading line
            for line in annotation_file:

                if len(line) == 0:
                    continue

                if 'NAME' in line:
                    object_name = line.strip().split(':', 1)[-1]

                    if object_name not in self.objects.keys():
                        self.objects[object_name] = dict()
                        self.objects[object_name]['boxes'] = []
                        self.objects[object_name]['frames'] = []

                    continue

                if 'RECT' in line:
                    frame = line.strip().split(None, 5)
                    frame = [s.replace(',', '') for s in frame]
                    frame_idx = int(frame[1])
                    bb = [int(frame[p]) for p in list(range(2, 6))]

                    # if self.width_height is not None:
                    #     bb = clip_box_to_image(bb, self.width_height[1], self.width_height[0])

                    self.annotation_dict['frame_{:05d}'.format(frame_idx)][object_name] = bb
                    self.objects[object_name]['boxes'].append(bb)
                    self.objects[object_name]['frames'].append(frame_idx)

                    continue
        annotation_file.close()

        self.parsed = True
        self.error = False

        return True

    def is_valid(self):
        if self.parsed is False:
            return self._parse_file()
        else:
            return self.error

    def get_annoted_frame(self, frame_idx):
        return self.annotation_dict['frame_{:05d}'.format(frame_idx)]

    def filter_objects(self, objects):

        return {
            obj_tag: bb
            for obj_tag, bb in self.objects.items() if obj_tag.split('-')[0] in objects
        }


class AnnotationImage(object):
    def __init__(self, frame_number, annotation_path=None, encoding='ISO-8859-1'):
        # self.total_frames = total_frames
        self.frame_number = frame_number
        self.annotation_path = annotation_path
        self.encoding = encoding
        self.annotation_dict = {}
        self.parsed = False
        self.error = False

    def _parse_file(self):
        if (self.annotation_path is None) or (os.path.exists(self.annotation_path) is False):
            return False

        # create dictonary with number of frames
        # self.annotation_dict = {'frame_{:05d}'.format(d): {} for d in range(self.total_frames)}

        # reading annotation file
        with open(self.annotation_path, encoding=self.encoding) as annotation_file:

            # reading line
            for line in annotation_file:

                if len(line) == 0:
                    continue

                if 'NAME' in line:
                    object_name = line.strip().split(':', 1)[-1]
                    continue

                if 'RECT' in line:
                    frame = line.strip().split(None, 5)
                    frame = [s.replace(',', '') for s in frame]
                    frame_idx = int(frame[1])

                    # print(type(frame_idx))
                    # print(self.frame_number)
                    if frame_idx == self.frame_number:
                        bb = [int(frame[p]) for p in list(range(2, 6))]
                        self.annotation_dict['{}'.format(object_name)] = bb
                    continue

        annotation_file.close()

        self.parsed = True
        self.error = False

        return True

    def is_valid(self):
        if self.parsed is False:
            return self._parse_file()
        else:
            return self.error

    def parse_annotation(self):
        return self._parse_file()

    def get_annotations(self):
        if self.parsed is False:
            self.parse_annotation()
        return self.annotation_dict

    def get_bboxes_labels(self):
        annotation_dict = self.get_annotations()
        bboxes = [bbox for bbox in annotation_dict.values()]
        labels = [obj for obj in annotation_dict.keys()]
        return bboxes, labels
