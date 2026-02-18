from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyspectral-quantum-solver",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "sympy>=1.12",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "scipy>=1.11",
    ],
    python_requires=">=3.8",  
    author="Vagner Jandre Monteiro",
    author_email="vagner.jandre@iprj.uerj.br", 
    description="Spectral methods for solving eigenvalue problems in quantum wells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Legeandre/PySpectral-Quantum-Solver",
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
