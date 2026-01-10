# Archx
A cost modeling framework to explore the system design space based on A-Graph.

## Installation
All provided installation methods allow running ```archx``` in the command line and ```import archx``` as a python module.

Make sure you have [Anaconda](https://www.anaconda.com/) installed before the steps below.

### AE setup
This is the developer mode, where you can edit the source code with live changes reflected for simulation.
1. ```git clone``` [this repo](https://github.com/UnaryLab/archx) and ```cd archx``` to the repo dir.
2. ```git fetch --all` to retrieve branches.
3. ```git checkout -b asplos_2026_ae origin/asplos_2026_ae``` to switch to the AE branch
4. ```conda env create -f environment.yaml```
   - The ```name: archx``` in ```evironment.yaml``` can be updated to a preferred one.
5. ```conda activate archx```
6. Validate installation via ```archx -h``` in the command line or ```import archx``` in python code.
7. ```bash run_mugi.sh``` to run the simulation workflow.
8. Output figures can be found in ```zoo/llm/results/figs/``` and ```zoo/llm/results/tables```.

## Citation
If Mugi has been useful in your own research, please cite us using the following bibtex citation:

```
@inproceedings{price2026asplos,
  title     = {Mugi: Value Level Parallelism For Efficient LLMs},
  author    = {Daniel Price and Prabhu Vellaisamy and John Paul Shen and Di Wu},
  booktitle = {International Conference on Architectural Support for Programming Languages and Operating Systems},
  year      = {2026}
}
```
