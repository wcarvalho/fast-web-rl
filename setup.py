from setuptools import setup, find_packages

setup(
    name="FastWebRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "python-fasthtml",
        "fastapi"
    ],
    author="Wilka Carvalho",
    author_email="wcarvalho92@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcarvalho/fast-web-rl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
