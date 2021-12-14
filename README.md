# Quantum Monte Carlo simulations in the trimer basis: first-order transitions and thermal critical points in frustrated trilayer magnets

This repository contains the code and data used to create figures 4-10 in our submission [
Quantum Monte Carlo simulations in the trimer basis: first-order transitions and thermal critical points in frustrated trilayer magnets](https://scipost.org/submissions/2105.05271v4/).

## Dependencies
* Python 3
* Matplotlib >=3.5.1
* Numpy >=1.21.4
* Scipy >=1.7.3

## Usage

Run

```
	python3 <script>
```

from within the `scripts` directory.

## Contents
* `scripts/finite_size_scaling.py`
	+ `plots/critpoint.pdf`: Fig. 5
	+ `plots/specheat.pdf`: Fig. 8
	+ `plots/corrlen_scaling.pdf`: Fig. R2
* `scripts/ising.py`
	+ `plots/ising_rotation.pdf`: Fig. 7
* `scripts/ising_critical.py`
	+ `plots/ising_critical.pdf`: Fig. R1
* `scripts/phasediag.py`
	+ `plots/phasediag.pdf`: Fig. 4
	+ `plots/specheat_scan.pdf`: Fig. 6
	+ `plots/corrlen.pdf`: Fig. 9
* `scripts/spin32energy.py`:
	+ `plots/spin32.pdf`: Fig. 10

Figs. R1 and R2 are part of our [Response to Referee #1](https://scipost.org/submissions/2105.05271v2/#comment_id1678).

