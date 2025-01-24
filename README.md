# Flow Matching using `tinygrad`

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

[ ] more path objects
[ ] non-euclidean flow matching
[ ] discrete flow matching

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
