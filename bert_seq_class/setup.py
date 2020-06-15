import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bert_seq_model",
    version="0.0.1",
    author="Jerry Xu",
    author_email="jerry.xu@csiro.au",
    description="BERT model for sequence classification implemented using HuggingFace and Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.csiro.au/users/xu081/repos/nlp_in_ir/browse",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)