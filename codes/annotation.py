import os


class Annotation:
    def __init__(self, annotation_path=None, total_frames=None):
        self.total_frames = total_frames
        self.annotation_path = annotation_path
        self.annotation_dict = {}
        self.parsed = False
        self.error = False

    def _parse_file(self):
        if (self.annotation_path is None) or (os.path.exists(self.annotation_path) is False):
            return False

        # create dictonary with number of frames
        self.annotation_dict = {'frame_{:04d}'.format(d): {} for d in range(self.total_frames)}

        # reading annotation file
        with open(self.annotation_path) as annotation_file:

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
                    bb = [int(frame[p]) for p in list(range(2, 6))]

                    self.annotation_dict['frame_{:04d}'.format(frame_idx)][object_name] = bb
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
        return self.annotation_dict['frame_{:04d}'.format(frame_idx)]
