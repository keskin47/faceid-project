from setuptools import setup, find_packages

setup(
    name="faceid",
    version="1.0.0",
    author="Bünyamin Keskin",
    description="FaceID-based automated attendance system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "opencv-python",
        "numpy",
        "insightface",
        "PyYAML",
        "pandas",
        "openpyxl",
    ],
)
