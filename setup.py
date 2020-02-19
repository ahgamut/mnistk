from setuptools import setup, find_packages

with open("README.md", "r") as helpfile:
    readme_txt = helpfile.read()

setup(
    name="mnistk",
    version="0.3.1",
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    description="neural nets on the mnist dataset",
    long_description=readme_txt,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.17.2",
        "matplotlib==3.0.3",
        "click==7.0",
        "torch==1.3.0",
        "torchvision==0.4.1",
        "scikit-learn==0.20.3",
        "h5py==2.10.0",
        "pandas==0.24.2",
        "Pillow==5.4.1",
        "torchrec",
    ],
    url="https://github.com/ahgamut/mnistk",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
