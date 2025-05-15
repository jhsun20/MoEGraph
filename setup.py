import os
import subprocess
from setuptools import setup, find_packages

# Clone the GOOD repository if it doesn't exist
if not os.path.exists('GOOD'):
    subprocess.call(['git', 'clone', 'https://github.com/divelab/GOOD.git'])
    # Install the GOOD package
    subprocess.call(['pip', 'install', '-e', './GOOD'])

setup(
    name="graph_moe_ood",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
) 