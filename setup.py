from setuptools import setup, find_packages
# 暂时还不执行
# 创建 setup.py 文件（内容如下），用于将项目标记为 Python 包：
setup(
    name="easyedit_project",  # 包名自定义
    version="0.1",
    packages=find_packages(),  # 自动识别easyeditor等子包
    zip_safe=False,
)
# 在项目根目录执行以下命令，将项目安装为 Python 可编辑包（修改代码无需重新安装）：
##pip install -e . 
# # 项目根目录执行（注意：脚本名不带.py）
# 不再直接运行 python train_dental_lora2.py，而是通过 python -m 以「模块模式」运行（此时脚本属于包的一部分，支持相对导入）：
# python -m train_dental_lora2


#相对导入（from .xxx import ...）仅适用于 Python 包内部的模块之间，而直接运行项目根目录的 train_dental_lora2.py 时，
#该脚本会被 Python 识别为「顶级脚本」（__name__ == "__main__"），没有 “父包” 概念，因此 from .easyeditor import ... 
#这种相对导入会触发 ImportError。