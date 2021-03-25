import os

def remove_ext(filepath):
    """Removes all extensions of the filename pointed by filepath.

    :Example:

    >>> remove_ext('home/user/t1_image.nii.gz')
    'home/user/t1_image'
    """
    paths = filepath.split('/')
    return filepath if '.' not in paths[-1] else os.sep.join(paths[:-1] + [paths[-1].split('.')[0]])


def make_dirs(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_filename(filepath, ext=True):
    """Returns the filename of the file pointed by `filepath`.

    :Example:

    >>> get_filename('/home/user/t1_image.nii.gz')
    't1_image.nii.gz'
    >>> get_filename('/home/user/t1_image.nii.gz', ext=False)
    't1_image'
    """
    filename = os.path.basename(filepath)
    return filename if ext else filename.split('.')[0]


def get_path(filepath):
    """Returns the base path of the file pointed by `filepath`.

    :Example:

    >>> get_path('/home/user/t1_image.nii.gz')
    '/home/user'
    """
    return os.path.dirname(filepath)


if __name__ == '__main__':
    print(remove_ext('/home/albert/Downloads/AIIM-D-20-00213_reviewer.pdf'))
    print(remove_ext('/home/albert/Downloads/AIIM-D-20-00213_reviewer'))