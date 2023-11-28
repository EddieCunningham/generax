import setuptools

setuptools.setup(
    name="generax",
    version="0.0.5",
    author="Edmond Cunningham",
    author_email="edmondcunnin@cs.umass.edu",
    description="Generative Models using Jax",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EddieCunningham/generax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().splitlines(),
)