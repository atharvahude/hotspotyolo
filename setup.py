from setuptools import setup, find_packages

setup(
    name='hotspotyolo',
    version='0.1.0',
    author='Atharva Hude',
    author_email='ahude@asu.edu',
    description='A package for generating CAMs using the YOLOv8 - v11 models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'opencv-python',
        'numpy',
        'matplotlib',
        'Pillow',
        'tqdm',
        'ultralytics',
        'pytorch-grad-cam',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)