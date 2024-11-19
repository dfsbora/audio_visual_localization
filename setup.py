from setuptools import setup, find_packages

def parse_requirements(filename):
    """ Load requirements from a requirements file. """
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Path to your requirements.txt
requirements_path = 'requirements.txt'

setup(
    name='audio_visual_localization',
    version='0.1.0',
    #packages=['audio_visual_localization'],
    packages=find_packages(where='src'),
    install_requires=parse_requirements(requirements_path),
    package_dir={'': 'src'},
)


