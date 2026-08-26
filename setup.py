from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return a list of requirements from the given file path.
    '''
    requirements = []
    with open(file_path, encoding='utf-8-sig') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='cardio-disease-prediction',
    version='0.1',
    author='Omar Diop',
    author_email='od14034@gmail.com',
    description='Prédiction du risque de maladie cardiovasculaire à partir de données cliniques',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
)