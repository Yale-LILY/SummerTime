# InfiniBatch

To view the documentation, please clone the repository and go to docs/infinibatch/index.html

To run unit tests, run the following command.
```
python -m unittest discover -s test
```

When working on the documentation, install pdoc:
```
pip install pdoc3
```
You can then start a local http server that dynamically updates the documentation:
```
pdoc --template-dir docs --http : infinibatch
```

We currently haven't set up the CI to automatically generate the documentation.
Before you merge anything into master, please delete the existing documentation in docs/infinibatch and run
```
pdoc -o docs --template-dir docs --html infinibatch
```