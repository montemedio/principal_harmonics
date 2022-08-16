from setuptools import setup

setup(
    name='principal_harmonics',
    version='0.1.0',    
    description='A example Python package',
    author='Lukas Middelberg',
    author_email='lukas@montemedio.de',
    packages=['principal_harmonics'],
    install_requires=['pya', 'librosa', 'matplotlib', 'scipy', 'numpy', 'mido', 'sklearn', 'ipympl', 'pandas', 'tqdm',

        # For some reason pip complains that these packages are not installed...
        # They seem to be some kind of transitive dependency.
        'zlmdb', 'treq'
    ],
)
