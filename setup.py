import setuptools, os

PACKAGE_NAME = 'color_summary'
VERSION = '1.0'
AUTHOR = 'wgs'
EMAIL = '1151573613@qq.com'
DESCRIPTION = '图片色系的提取与概述'
GITHUB_URL = 'https://github.com/WGS-note/color_summary'

parent_dir = os.path.dirname(os.path.realpath(__file__))
import_name = os.path.basename(parent_dir)

with open('{}/README.md'.format(parent_dir), 'r') as f:
    long_description = f.read()

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=GITHUB_URL,
    packages=[
        'color_summary',
        'color_summary.assets',
    ],
    package_dir={'color_summary':'.'},
    package_data={'': ['*.csv', '*.jpg', '*.png']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        'scipy',
        'opencv-python>=4.7.0.72',
    ],
)
