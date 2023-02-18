# UMPIRE

![Actions tests.yml](https://github.com/DiracKeteering/UMPIRE/actions/workflows/tests.yml/badge.svg)

Implementation of the UMPIRE algorithm from the [following paper](https://onlinelibrary.wiley.com/doi/10.1002/mrm.24897):

> Simon Robinson, Horst Schödl, Siegfried Trattnig, 2014 July,
> "A method for unwrapping highly wrapped multi-echo phase images at
> very high field: UMPIRE", Magnetic Resonance in Medicine, 72(1):80-92
> DOI: 10.1002/mrm.24897

## Installation -- Usage

```bash
$ git clone https://github.com/DiracKeteering/UMPIRE.git

$ cd UMPIRE

$ pip install -e .
```

## Installation -- Development
```bash
$ git clone https://github.com/DiracKeteering/UMPIRE.git

$ cd UMPIRE

$ pip install -e ".[dev]"
```

### Adding Dependencies

All package handling is done inside the `pyproject.toml` file. Here are the steps necessary to add dependencies to the package:

0. Before adding a dependency to the package, make sure it is compatible! (See **Tox**!)

1. Open the file `pyproject.toml` and go to `[project]`. Then, add the dependency inside the `dependencies`-list. You can use the syntax [listed here](https://python-poetry.org/docs/dependency-specification/) to specify the dependencies.

2. Use the `pip-tools` command to compile dependencies files:

- Generate usage dependencies file:  

```bash
$ pip-compile --output-file=requirements.txt --resolver=backtracking
```

- Generate development dependencies file:

```bash
$ pip-compile --extra=dev --output-file=dev-requirements.txt --resolver=backtracking pyproject.toml
```