"""
Written by
Mauricio Abello
"""

from os import listdir
from os.path import isfile, join


class FileHandler:

    def __init__(self):
        pass

    @staticmethod
    def get_all_files(dir_path):

        files = []
        for f in listdir(dir_path):
            full_name = join(dir_path, f)
            if isfile(full_name):
                files.append(full_name)

        return files

    @staticmethod
    def join_paths(path1, path2):
        return join(path1, path2)

