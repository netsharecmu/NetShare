from pathlib import Path

from setuptools import find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "NetShare"
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="netshare",
    version=VERSION,
    author="Yucheng Yin, Zinan Lin, Minhao Jin, Giulia Fanti, Vyas Sekar, Saar Tochner",
    author_email="yyin4@andrew.cmu.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    extras_require={
        "dev": [
            "pytest==7.0.1",
            "moto==4.0.12",
            "pre-commit==2.17.0",
            "mypy==0.971",
            "pandas-stubs==1.2.0.62",
            "boto3-stubs==1.24.35",
            "tensor-annotations-tensorflow-stubs==2.0.2",
            "types-setuptools==65.6.0.2",
        ],
        "aws": [
            "boto3==1.26.33",
        ],
    },
    install_requires=[
        "tensorflow==1.15",
        "tensorflow-privacy==0.5.0",
        "tqdm",
        "matplotlib",
        "pandas",
        "sklearn",
        "more-itertools",
        "gensim==3.8.3",
        "networkx",
        "notebook",
        "ipyplot",
        "jupyterlab",
        "statsmodels",
        "gdown",
        "annoy",
        "pyshark",
        "scapy",
        "ray",
        "ray[default]",
        "multiprocess",
        "addict",
        "config_io==0.4.0",
        "flask",
        "scikit-learn==0.24.2",
        "cmake==3.25.0",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "netshare"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.6",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
