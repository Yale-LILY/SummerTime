import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="SummerTime",
    version="0.1",
    author="Ansong Ni, Murori Mutuma, Zhangir Azerbayev, Yusen Zhang, Tao Yu, Dragomir Radev",
    author_email="ansong.ni@yale.edu, dragomir.radev@yale.edu",
    description="Text summarization for non-experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yale-LILY/SummerTime",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
