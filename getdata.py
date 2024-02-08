import requests
import os
import zipfile

# URL列表
urls = [
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_202002.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201902.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201609.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201402.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_201105.zip",
    "https://media.fdj.fr/static-draws/csv/euromillions/euromillions_200402.zip"
]

# 存储路径
save_path = '/Users/dantashashou/Downloads/euromillions'  # 请根据实际情况调整路径

# 检查目标目录是否存在，如果不存在，则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 下载并解压文件
for url in urls:
    file_name = url.split('/')[-1]
    full_path = os.path.join(save_path, file_name)

    # 发起请求下载文件
    response = requests.get(url)
    if response.status_code == 200:
        # 写入文件到指定路径
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f"文件已下载并保存到：{full_path}")
        
        # 解压文件
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)
        print(f"文件已解压到：{save_path}")
        
        # 删除原zip文件
        os.remove(full_path)
        print(f"原zip文件已删除：{full_path}")
    else:
        print(f"下载失败，状态码：{response.status_code}")
