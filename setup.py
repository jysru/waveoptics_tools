import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="waveoptics",
    version="0.1",
    author="Jysru",
    author_email="jysru@pm.me",
    description="Some tools for wave optics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jysru/waveoptics_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)