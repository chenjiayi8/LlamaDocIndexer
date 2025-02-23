from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="LlamaDocIndexer",
    version="0.3.2",
    packages=find_packages(),
    description="A tool to index and monitor changes in document directories",
    long_description_content_type="text/markdown",
    author="Jiayi Chen",
    author_email="chenjiayi_344@hotmail.com",
    url="https://github.com/chenjiayi8/LlamaDocIndexer",
    download_url="https://pypi.org/project/LlamaDocIndexer/",
    install_requires=required,
)
