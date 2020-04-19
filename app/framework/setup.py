# Dmitry Kisler © 2020-present
# www.dkisler.com

import pathlib
from setuptools import setup, find_packages, find_namespace_packages


DIR = pathlib.Path(__file__).parent
requirements = (DIR / "requirements.txt").read_text()
README = (DIR / "README.md").read_text()

setup(
    name="pos_tagger_framework",
    version='1.0.0',
    description="Framework to develop PoS tagger models.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/kislerdm/assessment_back_ml_eng",
    author="Dmitry Kisler",
    author_email="admin@dkisler.com",
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_namespace_packages(where=".",
                                     exclude=("tests",)),
    install_requires=requirements,
    include_package_data=True,
)
