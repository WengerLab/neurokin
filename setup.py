from setuptools import setup, find_packages

setup(
    name='neurokin',
    version='0.0.1',
    description='package to support neural and data analysis',
    url='https://github.com/WengerLab/neurokin',
    author='Elisa L. Garulli',
    author_email='e.garulli@charite.de',
    license='BSD 2-clause',
    packages=find_packages('neurokin'),
    package_dir={'': 'neurokin'},
    install_requires=['c3d==0.5.1',
                      'dlc2kinematics',
                      'fooof==1.0.0',
                      'mock==4.0.3',
                      'munkres==1.1.4',
                      'tdt==0.5.3'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)