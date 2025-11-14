from pathlib import Path
import glob

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """

    def __init__(
        self,
        root='',
        data_dir="data",
        dataset_dir="MSMT17",
        verbose=True,
        pid_begin=0,
        **kwargs
    ):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = Path(root) / data_dir / dataset_dir
        self.train_dir = self.dataset_dir / "bounding_box_train"
        self.query_dir = self.dataset_dir / "query"
        self.gallery_dir = self.dataset_dir / "bounding_box_test"

        self._check_before_run()
        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not self.dataset_dir.exists():
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not self.train_dir.exists():
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not self.query_dir.exists():
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not self.gallery_dir.exists():
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path):
        img_paths = glob.glob(str(Path(dir_path) / "*.jpg"))

        dataset = []
        pid_container = set()
        cam_container = set()
        for img_path in sorted(img_paths):
            path = Path(img_path)
            pid, camid, _ = path.stem.split('_')
            pid = int(pid)  # no need to relabel
            camid = int(camid[1:])
            dataset.append((img_path, self.pid_begin+pid, camid-1, 0))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset