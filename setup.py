"""
Setup script for GT - Distributed GPU ML Operations.
"""

from setuptools import setup, find_packages
import os


# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


# Read README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, encoding='utf-8') as f:
        return f.read()


setup(
    name='gt',
    version='0.1.0',
    description='Distributed frontend for GPU ML operations',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='GT Team',
    author_email='',
    url='https://github.com/yourusername/gt2',
    packages=find_packages(exclude=['tests', 'benchmarks', 'examples']),
    python_requires='>=3.8',
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'gt-server=gt.server.__main__:main',
            'gt-worker=gt.worker.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='distributed machine-learning gpu pytorch tensor',
    project_urls={
        'Documentation': 'https://github.com/yourusername/gt2',
        'Source': 'https://github.com/yourusername/gt2',
        'Bug Reports': 'https://github.com/yourusername/gt2/issues',
    },
)
