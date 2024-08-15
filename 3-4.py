#通过api下载数据集1
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate() # 获取 Kaggle API 的认证信息
# 下载 Titanic 数据集
api.dataset_download_files("heptapod/titanic")

#通过api下载数据集2
#kaggle competitions download -c titanic

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import zipfile  # 用于处理ZIP文件


# 定义解压ZIP文件的函数
def unzip_file(zip_path, extract_path=None):
    """
    解压ZIP文件到指定目录。

    :param zip_path: ZIP文件的路径。
    :param extract_path: 解压到的目标文件夹路径，默认为ZIP文件所在目录去掉.zip后缀。
    """
    if extract_path is None:
        # 如果没有指定解压路径，则默认使用ZIP文件路径去掉.zip后缀作为解压目录
        extract_path = zip_path[:-4]

        # 使用with语句确保ZipFile对象正确关闭
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 解压ZIP文件到指定目录
        zip_ref.extractall(extract_path)
        print(f"文件已解压到 {extract_path}")

    # ZIP文件的路径（这里假设titanic.zip已经存在于当前目录下）


zip_file_path = 'titanic.zip'
# 调用解压函数，解压到'titanic'文件夹
unzip_file(zip_file_path, 'titanic')

# 加载解压后的CSV文件
titanic_df = pd.read_csv("titanic/train.csv")  # 假设解压后的CSV文件名为train.csv且位于'titanic'文件夹内

# 处理Age列的缺失值，使用Age列的中位数填充
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# 处理Cabin列的缺失值，将缺失值替换为'Unknown'
titanic_df['Cabin'].fillna('Unknown', inplace=True)

# 打印处理后的部分表格，包括PassengerId、Age和Cabin这三列
print(titanic_df[['PassengerId', 'Age', 'Cabin']].head())  # 使用head()方法打印前5行数据

