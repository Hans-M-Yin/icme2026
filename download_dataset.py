from huggingface_hub import hf_hub_download

repo_id = "huanngzh/3D-Front"
repo_type = "dataset"
download_dir = "./MIDI-3D-committed-files/3d_front"  # 指定下载目录

file_list = [
    "3D-FRONT-TEST-RENDER.tar.gz",
    "3D-FRONT-TEST-SCENE.tar.gz",
    "3D-FRONT-TEST-SURFACE.partaa",
    "3D-FRONT-TEST-SURFACE.partab",
    "3D-FRONT-TEST-SURFACE.partac",
    "midi_test_furniture_ids.json",
    "midi_test_room_ids.json"
]

from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def download_file(filename):
    """单个文件下载函数"""
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=download_dir
        )
        return filename, file_path, None
    except Exception as e:
        return filename, None, str(e)


# 并行下载
def download_parallel(file_list, max_workers=4):
    downloaded_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_file = {
            executor.submit(download_file, filename): filename
            for filename in file_list
        }

        # 处理完成的任务
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                filename, file_path, error = future.result()
                if error:
                    print(f"❌ 下载失败 {filename}: {error}")
                else:
                    print(f"✅ 已下载: {filename}")
                    downloaded_files.append(file_path)
            except Exception as e:
                print(f"❌ 任务异常 {filename}: {e}")

    return downloaded_files


# 使用示例
downloaded_files = download_parallel(file_list, max_workers=4)