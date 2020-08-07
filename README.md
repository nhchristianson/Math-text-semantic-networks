# Semantic Networks in Mathematics Texts

This repository contains code associated with the paper

> Nicolas H. Christianson, Ann Sizemore Blevins, and Danielle S. Bassett (2020) <br/>
> [Architecture and evolution of semantic networks in mathematics texts](http://doi.org/10.1098/rspa.2019.0741) <br/>
> Proc. R. Soc. A. 476: 20190741.

In particular, it contains tools for the construction and analysis of semantic networks from mathematics textbooks, as well as Jupyter notebooks detailing the production of the results included in the aforementioned paper. The methodology is designed to be broadly applicable to any text, expository or otherwise, with some small modifications. If you find this code useful in your research, please consider citing our paper; a BibTeX citation is given in [`CITATION.bib`](CITATION.bib).

## Requirements

This code is tested on Python 3.7, but may work in other versions of Python 3.

A few Python packages are not strictly necessary for things to work, but may be useful:

- [``dionysus``](https://github.com/mrzv/dionysus), a persistent homology package. This is not the package we generally use for calculating the persistent homology of growing semantic networks (which is [``ripser.py``](https://ripser.scikit-tda.org/)), but it may be useful in conjunction with:
- [``cyclonysus``](https://github.com/sauln/cyclonysus), which wraps ``dionysus`` and enables the extraction of representative cycles from persistent homology, useful for visualizing the "knowledge gaps" extracted from a growing semantic network.
- [``graph-tool``](https://graph-tool.skewed.de/), which enables construction and analysis of graphs and much more; our paper's results use its graph plotting functionality.

All other necessary packages will be installed as dependencies by pip.

## Installation
Download this repository, navigate into the folder, and run ``pip install .``.

## Examples
The [`Notebooks`](Notebooks) folder contains as examples the Jupyter notebooks used to generate the results from our paper.

## License
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
