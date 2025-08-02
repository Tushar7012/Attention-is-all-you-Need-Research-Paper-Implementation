from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "Attention-is-all-you-Need-Research-Paper-Implementation"
AUTHOR_USER_NAME = "Tushar7012"
SRC_REPO = "TransFormer"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Tushar Das",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="td220627@gmail.com",
    packages=find_packages(),
    license="MIT",
    install_requires=LIST_OF_REQUIREMENTS
)