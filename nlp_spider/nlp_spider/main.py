from scrapy.cmdline import  execute


import sys
import os

# 进行运行目录的基本设置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 现在执行这个文件,传入的是一个列表["scrapy", "crawl", "刚刚自己设定的文件名"]
execute(["scrapy", "crawl", "granbluefansty"])