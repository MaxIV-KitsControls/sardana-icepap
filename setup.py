from setuptools import setup, find_packages

# The version is updated automatically with bumpversion
# Do not update manually
__version = "2.4.0"


setup(
    name="sardana_icepap",
    version=__version,
    packages=find_packages(),
    install_requires=["sardana", "icepap", "setuptools"],
    description="IcePAP Sardana controller."
)
