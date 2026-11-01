import os
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Lit le fichier de dépendances requirements.txt et extrait la liste des packages.
    """
    requirements = []
    with open(file_path, encoding='utf-8') as file_obj:
        requirements = [line.strip() for line in file_obj if line.strip() and not line.strip().startswith('#')]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# Lecture de la description longue depuis README.md
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='cardio-disease-prediction',
    version='0.1.0',
    author='Omar Diop',
    author_email='od14034@gmail.com',
    description="Système Expert d'Aide à la Décision Clinique pour la Prédiction du Risque Cardiovasculaire",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/omar-034/Cardiovascular_Disease_ML_Project',
    license='MIT',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)
