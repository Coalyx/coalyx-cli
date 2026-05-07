from setuptools import setup, find_packages

setup(
    name="coalyx-cli",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.34.0",
        "pydantic>=2.5.0",
        "rich>=13.7.0",
        "typer>=0.9.0",
        "google-generativeai>=0.4.0",
        "numpy>=1.26.0",
        "psutil>=5.9.0",
        "prompt_toolkit>=3.0.0"
    ],
    entry_points={
        "console_scripts": [
            "coalyx=src.cli.main:app",
        ],
    },
)
