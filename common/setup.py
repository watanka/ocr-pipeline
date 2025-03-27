from setuptools import setup, find_packages

setup(
    name="ocr-common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="OCR 서비스들의 공통 모델 및 유틸리티",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 