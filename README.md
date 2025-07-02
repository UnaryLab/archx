# Archx
A cost modeling framework to explore the system design space based on A-Graph.

## Installation
All provided installation methods allow running ```archx``` in the command line and ```import archx``` as a python module.

Make sure you have [Anaconda](https://www.anaconda.com/) installed before the steps below.

### Option 1: pip installation
1. ```conda env create -f environment.yaml```
   - The ```name: archx``` in ```evironment.yaml``` can be updated to a preferred one.
2. ```conda activate archx```
3. ```pip install archx```
4. Validate installation via ```archx -h``` in the command line or ```import archx``` in python code

### Option 2: source installation
This is the developer mode, where you can edit the source code with live changes reflected for simulation.
1. ```git clone``` [this repo](https://github.com/UnaryLab/archx) and ```cd``` to the repo dir
2. ```conda env create -f environment.yaml```
3. ```conda activate archx```
4. ```python3 -m pip install -e . --no-deps```
5. Validate installation via ```archx -h``` in the command line or ```import archx``` in python code
