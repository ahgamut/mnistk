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
        "click==7.0",
        "scikit-learn==0.20.3",
        "h5py==2.10.0",
        "pandas==0.24.2",
        "Pillow==8.3.2",
        "sqlalchemy==1.3.1",
        "torch>=1.3.0",
        "torchvision>=0.4.1",
        "torchrecorder",
    ],
    url="https://github.com/ahgamut/mnistk",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
