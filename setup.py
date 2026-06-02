from setuptools import setup, find_packages

setup(
    name="sentiment-cross-domain",
    version="0.1.0",
    description="Cross-Domain Sentiment Analysis with KL Domain Alignment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "hydra-core>=1.3.0",
        "wandb>=0.17.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pillow>=10.0.0",
        "reportlab>=4.0.0",
        "regex>=2024.4.0",
    ],
)
