from setuptools import setup, find_packages
from pathlib import Path
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if "-e ." in requirements:
            requirements.remove("-e .")


setup(
    name="personalmlproject",
    version="0.1.0",
    author="Debanjan",
    author_email="sarkardebanjan137@gmail.com",
    #url="https://example.com/personalmlproject",
    #license="MIT",
    packages=find_packages(),
    #include_package_data=True,
    python_requires=">=3.12",
    install_requires=get_requirements('requirements.txt'),
)