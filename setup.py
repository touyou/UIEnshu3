from setuptools import setup

setup(
    name='Photo2ClipArtSample',
    version='0.0.1',
    description='Sample implementation of Photo2ClipArt',
    long_description=readme,
    author='touyou',
    url='https://github.com/touyou/',
    install_requires=['numpy', 'cv2'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'old'))
)