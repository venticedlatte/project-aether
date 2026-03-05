from setuptools import setup, find_packages

setup(
    name="aether_engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "netket",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic"
    ],
)
