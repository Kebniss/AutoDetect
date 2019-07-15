import os
import ntpath


def get_filename_extension(path):
    basename = ntpath.basename(path).split('.')
    return basename[0], basename[1]


def mkdir(folder_path):
    # if folder_path does not exist create it 
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
