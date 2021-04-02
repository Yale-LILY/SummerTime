import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='SummerTime',  
     version='0.1',
     scripts=['summertime'] ,
     author="Murori Mutuma, Zhangir",
     author_email="murorimutuma@gmail.com",
     description="A summarization mode",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/LILYlab",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
