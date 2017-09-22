"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
import os
from codecs import open


about = {}
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'ptutils', '__version__.py'), 'r', 'utf-8') as f:
    exec(f.read(), about)

# Get the long description from the README file
with open('README.md', 'r', 'utf-8') as f:
    long_description = f.read()

tests_require = ['nose']
requires = ['numpy', 'torch', 'pymongo', 'gitpython']
packages = find_packages(exclude=['contrib', 'docs', 'tests'])

setup(
    url=about['__url__'],
    name=about['__title__'],
    author=about['__author__'],
    license=about['__license__'],
    version=about['__version__'],
    keywords=about['__keywords__'],
    description=about['__description__'],
    long_description=long_description,
    tests_require=tests_require,
    install_requires=requires,
    packages=packages,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
