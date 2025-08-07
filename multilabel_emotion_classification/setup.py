"""Setup script for multilabel emotion classification package."""

from setuptools import setup, find_packages
import os

# Read README.md
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() 
                   if line.strip() and not line.startswith("#")]
    return []

setup(
    name="multilabel-emotion-classification",
    version="1.0.0",
    author="Marileni Sinioraki",
    author_email="marileni.sinioraki@example.com",
    description="A modular implementation for multilabel emotion classification using BERT",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/marileni/multilabel-emotion-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "llm": [
            "openai>=0.27.0",
            "anthropic>=0.2.0",
        ],
        "viz": [
            "wandb>=0.12.0",
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-classify=multilabel_emotion_classification.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "multilabel_emotion_classification": [
            "config/*.json",
            "data/*.csv",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning", 
        "natural language processing",
        "emotion classification",
        "multilabel classification",
        "BERT",
        "transformers",
        "GoEmotions",
    ],
)
