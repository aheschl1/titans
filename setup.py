from setuptools import setup, find_packages

setup(
    name='titan',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchviz',
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
        ],
    },
    author='Andrew Heschl',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)