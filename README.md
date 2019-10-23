# `pynterp`: Interpretable Machine Learning in Python

The bottleneck to deploying machine learning, and therefore its use more broadly in a great range
of problems in business and society, is a lack of interpretability. This is the sentiment "we can't trust this black-box".
This repository aims to replicate various interpretable machine learning algorithms for their broad use and dissemination.

## Installation

It is highly recommended to use the provided [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment:
1. `conda env create -f environment.yml`
2. `conda activate pynterp`

To install, run:
1. `python setup.py install`

The basic interface is:
```python
from pynterp.rules.decision_set import DecisionSet
ds = DecisionSet()
ds.fit(X_train, y_train)
```

## Rules-Based Methods

Rules-based methods are models that learn a list or set of IF-THEN-ELSE rules;
they are highly interpretable and appealing to humans. The current implementation
of `DecisionSet` follows the [paper by Lakkaraju et. al](https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf).

## TODO

*October 23, 2019*:
- Implement test suite and continuous integration framework
- Incorporate Bayesian decision lists (due to Rudin et. al)
