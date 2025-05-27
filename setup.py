from setuptools import setup, find_packages

setup(
    name="minimal-text-diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "wandb",
        "tqdm",
        "blobfile",
    ],
) 