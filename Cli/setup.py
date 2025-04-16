"""
AIRA Connect - Universal Client for the AIRA Network
===================================================

A universal adapter library for connecting AI agents to the AIRA network,
regardless of the underlying framework (ADK, LangChain, etc.).
"""

from setuptools import setup, find_packages

# Read the long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = __doc__

setup(
    name="aira-connect",
    version="0.1.0",
    author="AIRA Team",
    author_email="contact@aira-network.org",
    description="Universal client for connecting AI agents to the AIRA network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aira-network/aira-connect",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "urllib3>=1.26.0",
    ],
    extras_require={
        "web": ["fastapi>=0.95.0", "uvicorn>=0.21.0"],
        "adk": ["google-adk>=0.1.0"],
        "langchain": ["langchain>=0.0.200"],
        "mcp": ["mcp>=0.1.0"],
        "all": [
            "fastapi>=0.95.0",
            "uvicorn>=0.21.0",
            "google-adk>=0.1.0",
            "langchain>=0.0.200",
            "mcp>=0.1.0",
        ],
    },
)