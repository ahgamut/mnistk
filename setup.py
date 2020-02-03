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
        "numpy",
        "matplotlib",
        "click",
        "torch",
        "torchvision",
        "scikit-learn",
        "h5py",
        "pandas",
        "torchrec",
    ],
    url="https://github.com/ahgamut/mnistk",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
