# Mugi: Value Level Parallelism for Efficient LLMs
Mugi introduces a single, cohesive architecture leveraging value-level parallelism for both
- general matrix multiplication (GEMM)
- nonlinear operations.

To learn more, feel free to read our [paper](https://dl.acm.org/doi/10.1145/3779212.3790189)

## Artifact Evaluation
> [!IMPORTANT]
> This evaluation is built using the newest version of archx, causing some values to differ by inconsequential amounts. To see the results true to the published paper, please run the [zenodo artifact](#zenodo).

To run the artifact to reproduce our results, please first install Archx [here](../../README.md).

After installing, you can call the script from the [archx base directory](../)

```
bash zoo/mugi/script.sh
```

## Results
After running the script, figures can be found [here](results/figs/) and tables can be found [here](results/tables/).

## Zenodo
A zenodo submission exists at https://zenodo.org/records/18063514

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