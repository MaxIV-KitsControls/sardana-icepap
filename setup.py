from setuptools import setup, find_packages

setup(
    name="sardana-icepap",
    use_scm_version=True,
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=["sardana", "icepap"],
)
