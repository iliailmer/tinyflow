# Flow Matching using `tinygrad`

[![Tests](https://github.com/iliailmer/tinyflow/actions/workflows/tests.yml/badge.svg)](https://github.com/iliailmer/tinyflow/actions/workflows/tests.yml)
[![Lint](https://github.com/iliailmer/tinyflow/actions/workflows/lint.yml/badge.svg)](https://github.com/iliailmer/flow_matching_tinygrad/actions/workflows/lint.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<!--toc:start-->

- [Flow Matching using `tinygrad`](#flow-matching-using-tinygrad)
  - [Introduction](#introduction)
  - [References](#references)
  <!--toc:end-->

## Introduction

This project is my exploration into flow matching
algorithms using `tinygrad` library instead of traditional
`pytorch` implementations.
This is a learning experience for
understanding both the flow matching and the library.

## Running the Code

I highly recommend using [`uv` tool](https://github.com/astral-sh/uv) to run this project.

```python
uv run example.py
```

## To-Do

Time permitting, I plan to add the following implementations:

- [x] more path objects

- [ ] non-euclidean flow matching

- [ ] discrete flow matching

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
