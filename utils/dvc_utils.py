import os
import subprocess
from pathlib import Path

import fire


def download_dvc_data(
    data_dir="data", dataset_name="Market-1501-v15.09.15.dvc", remote=None
):
    """
    Downloads data from DVC remote storage using CLI commands.
    """
    try:
        print(f"Downloading data from DVC remote to {data_dir}...")

        current_dir = Path.cwd()

        if current_dir.name != "reid-task":
            os.chdir(current_dir / "reid-task")

        cmd = ["dvc", "pull"]
        target_path = Path(data_dir) / dataset_name
        cmd.append(str(target_path))

        if remote:
            cmd.extend(["-r", remote])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"DVC pull failed: {result.stderr}")
            return False

        print(f"DVC pull output: {result.stdout}")
        return True
    except Exception as e:
        print(f"Error downloading data from DVC: {e}")
        return False
    finally:
        if current_dir.name != "reid-task":
            os.chdir(current_dir)


if __name__ == "__main__":
    fire.Fire(download_dvc_data)
