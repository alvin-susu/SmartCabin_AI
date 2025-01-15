"""
通用工具
"""
import os


def load_sub_files(parent_path: str):
    """
    加载路径下的所有文件并返回绝对路径
    :param parent_path: 父目录
    :return: 绝对路径
    """

    files_list = []
    for root, dirs, files in os.walk(parent_path):
        abspath = os.path.abspath(root)
        all_path = os.path.join(abspath, *files)
        print(f"all_path: {all_path}")
        files_list.append(all_path)
    return files_list




if __name__ == "__main__":
    print(load_sub_files("../knowledge_data"))