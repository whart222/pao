*PAO DRAFT DOCUMENTATION*

[![Actions Status](https://github.com/pyomo-annex/pao/workflows/continuous-integration/github/pr/linux/badge.svg)](https://github.com/pyomo-annex/pao/actions)
[![codecov](https://codecov.io/gh/pyomo-annex/pao/branch/master/graph/badge.svg)](https://codecov.io/gh/pyomo-annex/pao)
[![Documentation Status](https://readthedocs.org/projects/pao/badge/?version=latest)](http://pao.readthedocs.org/en/latest/)

[![Pyomo Checks - GitHub Master](https://github.com/pyomo-annex/pao/workflows/pyomo-checks/master/badge.svg)](https://github.com/pyomo-annex/pao/actions)
[![Pyomo Checks - PyPI](https://github.com/pyomo-annex/pao/workflows/pyomo-checks/pypi/badge.svg)](https://github.com/pyomo-annex/pao/actions)
[![Pyomo Checks - Conda](https://github.com/pyomo-annex/pao/workflows/pyomo-checks/conda/badge.svg)](https://github.com/pyomo-annex/pao/actions)

[![GitHub contributors](https://img.shields.io/github/contributors/pyomo-annex/pao.svg)](https://github.com/pyomo-annex/pao/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/pyomo-annex/pao.svg?label=merged+PRs)](https://github.com/pyomo-annex/pao/pulls?q=is:pr+is:merged)
[![Issue stats](http://isitmaintained.com/badge/resolution/pyomo-annex/pao.svg)](http://isitmaintained.com/project/pyomo-annex/pao)
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)

# PAO Overview

PAO is a Python-based package for Adversarial Optimization.  PAO extends the modeling concepts in [Pyomo](https://github.com/Pyomo/pyomo) to enable the expression and solution of multi-level optimization problems. The goal of this package is to provide a general modeling and analysis capability, and application exemplars serve to illustrate PAO's general capabilities.

This package was derived from the capabilities in pyomo.bilevel and pyomo.dualize, which are now deprecated.

Pyomo is available under the BSD License, see the LICENSE.txt file.

### Installation

#### PyPI [![PyPI version](https://img.shields.io/pypi/v/pao.svg?maxAge=3600)](https://pypi.org/project/pao/) [![PyPI downloads](https://img.shields.io/pypi/dm/pao.svg?maxAge=21600)](https://pypistats.org/packages/pao)

    pip install pao
    
### Testing

Pyomo is currently tested with the following Python implementations:

* CPython: 3.6, 3.7, 3.8

Testing 

* pip install nose coverage

* Simple tests

  * nosetests .

* Tests with coverage

  * nosetests --with-xunit --with-coverage --cover-xml .
  * coverage report -m

### Tutorials and Examples

* TBD

### Getting Help

* [Add a Ticket](https://github.com/pyomo-annex/pao/issues/new)
* [Find a Ticket](https://github.com/pyomo-annex/pao/issues) and **Vote On It**!

### Developers

By contributing to this software project, you are agreeing to the following terms and conditions for your contributions:

1. You agree your contributions are submitted under the BSD license. 
2. You represent you are authorized to make the contributions and grant the license. If your employer has rights to intellectual property that includes your contributions, you represent that you have received permission to make contributions and grant the required license on behalf of that employer.


