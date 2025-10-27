import errno
import json
from pathlib import Path


def mkdir_if_missing(directory):
    directory_path = Path(directory)
    if not directory_path.exists():
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    path_obj = Path(path)
    isfile = path_obj.is_file()
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    fpath_obj = Path(fpath)
    mkdir_if_missing(fpath_obj.parent)
    with open(fpath_obj, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def get_img_name_for_captions(captions_map, img_filename):
    # Base
    if img_filename in captions_map:
        return img_filename

    # Try Market-1501 format
    modified_name = img_filename[:-7] + "_00.jpg"
    if modified_name in captions_map:
        return modified_name

    return None
