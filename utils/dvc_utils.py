import os
import os.path as osp
import subprocess

def download_dvc_data(data_dir="data", dataset_name="Market-1501-v15.09.15.dvc", remote=None):
    """
    Downloads data from DVC remote storage using CLI commands.
    """
    try:
        print(f"Downloading data from DVC remote to {data_dir}...")

        current_dir = os.getcwd()
        
        if osp.basename(current_dir) != "reid-task":
            os.chdir(osp.join(current_dir, "reid-task"))
        
        cmd = ["dvc", "pull"]
        target_path = osp.join(data_dir, dataset_name)
        cmd.append(target_path)

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
        if osp.basename(current_dir) != "reid-task":
            os.chdir(current_dir)

if __name__ == "__main__":
    download_dvc_data()
