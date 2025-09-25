"""
Setup configuration for logistic regression from scratch package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Logistic Regression Implementation from Scratch"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.21.0',
        'autograd>=1.3.0',
        'matplotlib>=3.5.0',
        'scikit-learn>=1.0.0',
        'pytest>=6.0.0',
        'pytest-cov>=3.0.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.950'
    ]

setup(
    name="logistic-regression-from-scratch",
    version="1.0.0",
    author="Muhammad Moiz",
    author_email="moiz@example.com",
    description="A comprehensive implementation of logistic regression algorithms from scratch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/moiz/logistic-regression-from-scratch",
    project_urls={
        "Bug Reports": "https://github.com/moiz/logistic-regression-from-scratch/issues",
        "Source": "https://github.com/moiz/logistic-regression-from-scratch",
        "Documentation": "https://github.com/moiz/logistic-regression-from-scratch#readme",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "logistic-regression-demo=logistic_regression_from_scratch.examples.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "logistic_regression_from_scratch": ["data/*.csv"],
    },
    zip_safe=False,
    keywords="machine-learning logistic-regression classification gradient-descent",
)
