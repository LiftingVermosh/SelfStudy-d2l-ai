import os
import requests

url = 'https://www.gutenberg.org/files/35/35-0.txt'
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    text = response.text
    
    # 确保数据目录存在
    os.makedirs('./data', exist_ok=True)
    
    file_name = './data/time-machine-data.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)
    print("下载成功！文件保存为:", file_name)
    print("文件大小:", os.path.getsize(file_name))
    print(f"文本行数:{len(text.split('\n'))}")
else:
    print(f"下载失败，状态码: {response.status_code}")
