import setuptools

pk = setuptools.find_packages()

setuptools.setup(
    name="torch_convspline",
    version="1",
    author="Jelmer Cnossen",
    author_email="j.cnossen@gmail.com",
    description="",
    long_description="2D and 3D catmull-rom spline models evaluated using pytorch conv2d",
 #   long_description_content_type="text/markdown",
    url="",
    packages=pk,
	data_files=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows"
    ],
	install_requires=[
		'numpy',
		'matplotlib',
		'tqdm'
	]#,    distclass=BinaryDistribution
)
