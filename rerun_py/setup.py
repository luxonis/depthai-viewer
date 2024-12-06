from setuptools import setup, find_packages

setup(
    name='dynamic_recalib',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,  # Include files from MANIFEST.in
    package_data={
        'rerun': ['packages/*'],  # Include all files in the 'packages' folder
    },
)
