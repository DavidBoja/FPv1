# FPv1: 3D registration benchmark dataset

<br> 

⚠️⚠️⚠️ A new version of the FAUST-partial benchmark has been introduced [here](https://github.com/DavidBoja/exhaustive-grid-search), so we denominate this one as FPv1. Please use the new FAUST-partial benchmark instead of this one. ⚠️⚠️⚠️

<br>

Code for creating the FAUST-partial (now called FPv1) 3D registration benchmark from the paper: [Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration](https://arxiv.org/abs/2211.16301) presented at the BMVC 2022 workshop (URCV 22). <br>
For the baseline implementation please check [this Github repo](https://github.com/DavidBoja/greedy-grid-search).

<p align="center">
  <img src="https://github.com/DavidBoja/FAUST-partial/blob/main/assets/FAUST-partial-teaser.png" width="1000">
</p>

The benchmark generation for a single scan from the FAUST training dataset [1] can be summarized as follows:
1. Make $xz$-plane the floor by translating the minimal bounding box point of the scan to the origin
2. Surround the scan with a regular icosahaedron. Each point of the icosahaedron acts as a viewpoint
3. For each viewpoint, create a partial point cloud using the hidden point removal algorithm [2]

Finally, for a pair of partial point clouds with the desired overalp, generate a random rotation from the desired rotation range and translation range.


## Creating the benchmark
This code has been tested with python 3.6.9.

To create the benchmark:
1. Create the python virual environment by running `/bin/sh create_env.sh`
2. Activate the created environemnt `source faust-partial-env/bin/activate`
3. Setup the paths and variables in `config.yaml` under the `CREATE-BENCHMARK` category
4. Run `python create-FAUST-partial.py`
5. The benchmark csv has been created in the `SAVE-TO` directory under the `BENCHMARK-CSV-NAME` name.


To use the same benchmark as in the paper "Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration" you can find it in `data/benchmark_ico12_overlap60plus_withT.csv`.

## Visualization

To visualize and example from the benchmark dataset:
1. Setup the paths and variables in `config.yaml` under the `VISUALIZE-BENCHMARK` category
2. Run `python visualize_benchmark.py`

This opens two open3d windows -- the first one showing the initial displacement to be registered, the second the ground truth alignament.

## Using the benchmark

We provide the`load_benchmark.py` script that can be used to iteratively load each example from the benchmark.

To use the benchmark:
1. Setup the paths and variables in `config.yaml` under the `LOAD-BENCHMARK` category
2. Run `python load_benchmark.py` script -- additionally modifying the script to your needs

We provide the `evaluate.py` script that can be used to evaluate your results. The assumption is that your results are stored in the same way the `data/benchmark_ico12_overlap60plus_withT.csv` file is.

To evaluate the RR, RRE and RTE measures on the benchmark:
1. Setup the paths and variables in `config.yaml` under the `EVALUATE` category
2. Run `python evaluate.py` script

## Notes on Icosahaedron

The regular icosahaedron has 12 vertices and 20 triangular faces. Each vertex lies on the unit sphere and acts as a viewpoint from which we generate the partial point cloud. To obtain a higher resolution of the viewpoints, we provide a splitting strategy for the icosahaedron. To split an icosahaedron, each edge is split to create an additional vertex. This vertex is then projected to the unit sphere.

We provide the `ICOSAHAEDRON-NR-DIVISIONS` parameter in `config.yaml` to denote the number of times we repeat the process of splitting edges. The final number of vertices and faces can be found as:

$$ FACES = 20 \times 4^N $$

$$ VERTICES = 12 + \frac{3}{2} \sum_{i=0}^{N-1} (20 \times 4^i) $$

where `N` is the number of splits -- same as `ICOSAHAEDRON-NR-DIVISIONS`.

## 3D Registration results

| Method                                                            | RR(\%)           | RRE(&deg;)    | RTE(cm)          |
|-------------------------------------------------------------------|------------------|------------------|------------------|
| FPFH-8M  [3]                    | 9.51             | 4.347            | 1.900            |
| SpinNet  [4]                                         | 42.46            | 3.105            | 1.670            |
| GeoTransformer [5]  | 56.15            | 2.423            | 1.581            |
| FCGF+PointDSC [6]                                      | 47.85            | 3.354            | 1.793            |
| FCGF+YOHO-O [7]                                           | 18.91            | 4.489            | 1.852            |
| FCGF+YOHO-C [7]                                            | 29.18            | 3.653            | 1.668            |
| DIP [8]                                                    | 54.81            | 4.058            | 2.052            |
| Baseline                                                          | $\mathbf{92.81}$ | $\mathbf{0.014}$ | $\mathbf{0.009}$ |

For more details please check our paper.

## Citation

If you use our work, please reference our paper

```
@inproceedings{Bojanić-BMVC22-workshop,
   title = {Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration},
   author = {Bojani\'{c}, David and Bartol, Kristijan and Forest, Josep and Gumhold, Stefan and Petkovi\'{c}, Tomislav and Pribani\'{c}, Tomislav},
   booktitle={BMVC 2022 Workshop Universal Representations for Computer Vision},
   year = {2022}
   url={https://openreview.net/forum?id=tJ5jWBIAbT}
}
```

## References
[1] [Bogo et al.: Faust: Dataset and evaluation for 3D mesh registration. CVPR 2014](https://files.is.tue.mpg.de/black/papers/FAUST2014.pdf) <br />
[2] [Katz et al.: Direct visibility of point sets. ACM Transactions on Graphics 2007](https://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf) <br />
[3] [Rusu et al.: Fast Point Feature Histograms (FPFH) for 3D registration. ICRA 2009](https://ieeexplore.ieee.org/document/5152473) <br />
[4] [Ao et al.: SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. CVPR 2021](https://arxiv.org/abs/2011.12149) <br />
[5] [Zheng et al.: Geometric Transformer for Fast and Robust Point Cloud Registration. CVPR 2022](https://arxiv.org/abs/2202.06688) <br />
[6] [Bai et al.: PointDSC: Robust Point Cloud Registration Using Deep Spatial Consistency. CVPR 2021](https://arxiv.org/abs/2103.05465) <br />
[7] [Wang et al: You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors. ACM Multimedia 2022](https://arxiv.org/abs/2109.00182) <br />
[8] [Poiesi et al.: Distinctive 3D local deep descriptors. Pattern Recognition 2021](https://arxiv.org/abs/2009.00258) <br />
