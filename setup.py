from setuptools import setup

helpfile = open("README.md", "r")
readme_txt = helpfile.read()
helpfile.close()

setup(
    name="mnistk",
    version="0.3.1",
    author="Gautham Venkatasubramanian",
    author_email="ahgamut@gmail.com",
    description="neural nets on the mnist dataset",
    long_description=readme_txt,
    install_requires=["numpy", "click", "torch", "torchvision"],
    url="https://github.com/ahgamut/mnist1000",
    package_dir={"": "src"},
)
