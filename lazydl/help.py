from rich.markdown import Markdown
from rich.console import Console


def help():
    # 读取readme.md文件
    with open("README.md", "r", encoding='utf-8') as f:
        readme = f.read()
    # 打印readme.md文件
    md = Markdown(readme)
    console = Console()
    console.print(md)