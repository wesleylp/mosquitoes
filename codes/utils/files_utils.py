import os


class Directory(object):
    @staticmethod
    def get_files(root_dir, ext, recursive=False):
        """Get files in the root_dir

        Arguments:
            root_dir {str} -- String directory to search for files.
            ext {tuple} -- Tuple with extensions desired.

        Keyword Arguments:
            recursive {bool} -- Flag indicating whether to search files recursively or not. (default: {False})

        Returns:
            list -- list containing files pathes.
        """

        files = []

        if recursive:

            for (dirpath, dirnames, filenames) in os.walk(root_dir):

                if len(filenames) == 0:
                    continue

                # getting only the  files with desired extensions
                [
                    files.append(os.path.join(dirpath, s)) for s in filenames
                    if s.lower().endswith(ext)
                ]
        else:
            [files.append(s) for s in os.listdir(root_dir) if s.lower().endswith(ext)]

        return files
