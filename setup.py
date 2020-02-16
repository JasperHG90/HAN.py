import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTorch-HAN",
    version="0.0.1",
    author="Jasper Ginn",
    author_email="jasperginn@gmail.com",
    description="Train a Hierarchical Attention Model (HAN) on textual data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="placeholder",
    packages=setuptools.find_packages(),
    install_requires=["requests", "PyTorch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={'src.data': ['*.json']},
    include_package_data=True,
    python_requires='>=3.6'
)
