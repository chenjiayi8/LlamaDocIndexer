from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()

setup(
    name="LlamaDocIndexer",
    version="0.0.5",
    packages=find_packages(),
    description="A tool to index and monitor changes in document directories",
    author="Jiayi Chen",
    author_email="chenjiayi_344@hotmail.com",
    url="https://github.com/chenjiayi8/LlamaDocIndexer",
    install_requires=required,
)
