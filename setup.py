from setuptools import setup, find_packages

# The version is updated automatically with bumpversion
# Do not update manually
__version = "1.3.2"


setup(
    name="sardana_icepap",
    version=__version,
    packages=find_packages(),
    install_requires=["sardana", "icepap", "setuptools"],
    description="IcePAP Sardana controller."
)
