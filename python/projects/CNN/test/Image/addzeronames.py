from tkinter import Tk
from tkinter import filedialog
from classutils import ClassUtils
import os

total_digits = 4


def main():
    print('Initializing main function')

    # Withdrawing TKInter
    Tk().withdraw()

    # Ask for directory
    init_dir = '/home/mauricio/CNN/Images'
    options = {'initialdir': init_dir}
    dir_name = filedialog.askdirectory(**options)

    if not dir_name:
        print('Directory not selected')
    else:
        for root, subdirs, files in os.walk(dir_name):
            for file in files:
                full_path = os.path.join(root, file)
                print('Processing {0}'.format(full_path))

                extension = ClassUtils.get_filename_extension(file)

                if extension != '.jpg':
                    print('Ignoring file no jpg {0}'.format(full_path))
                    continue

                name = ClassUtils.get_filename_no_extension(file)
                if len(name) < total_digits:
                    new_name = ''
                    for _ in range(total_digits - len(name)):
                        new_name += '0'
                    new_name += name
                    new_name += extension

                    new_full_path = os.path.join(root, new_name)

                    print('Rename file {0} to {1}'.format(full_path, new_full_path))
                    os.rename(full_path, new_full_path)

        print('Done!')


if __name__ == '__main__':
    main()
