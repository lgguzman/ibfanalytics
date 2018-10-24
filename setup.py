import setuptools
def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]
reqs = parse_requirements('requirements.txt')
requirements = [str(ir) for ir in reqs]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ibfanalytics",
    version="0.0.1",
    author="Luis Guzmán",
    author_email="lgguzman890414@gmail.com",
    description="This is the module to ibf analytics",
    long_description="This is the module to ibf analytics",
    long_description_content_type="text/markdown",
    url="https://github.com/sbxcloud/sbxcloudpython",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)