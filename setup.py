from setuptools import setup

setup(
    name='principal_harmonics',
    version='0.1.0',    
    description='A example Python package',
    author='Lukas Middelberg',
    author_email='lukas@montemedio.de',
    packages=['principal_harmonics'],
    install_requires=['pya', 'librosa', 'matplotlib', 'scipy', 'numpy', 'mido', 'sklearn'],
)