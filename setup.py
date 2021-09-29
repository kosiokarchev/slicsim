from setuptools import setup, find_packages

setup(
    name='snai-tossn',
    description='pytorch-based simulations of supernovae',
    version='0.1',
    packages=find_packages(include=["snai.tossn"]),
    install_requires=[
        'torch', 'frozendict'
    ],
)
