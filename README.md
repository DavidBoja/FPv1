# FAUST-partial: 3D registration benchmark dataset

Code for creating the FAUST-partial 3D registration benchmark from the paper: "Challenging the Universal Representation of Deep Models for 3D Point Cloud Registration" presented at the BMVC 2022 workshop (URCV 22).

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

## Notes on Icosahaedron

The regular icosahaedron has 12 vertices and 20 triangular faces. Each vertex lies on the unit sphere and acts as a viewpoint from which we generate the partial point cloud. To obtain a higher resolution of the viewpoints, we provide a splitting strategy for the icosahaedron. To split an icosahaedron, each edge is split to create an additional vertex. This vertex is then projected to the unit sphere.

We provide the `ICOSAHAEDRON-NR-DIVISIONS` parameter in `config.yaml` to denote the number of times we repeat the process of splitting edges. The final number of vertices and faces can be found as:

$$ FACES = 20 \times 4^N $$

$$ VERTICES = 12 + \frac{3}{2} \sum_{i=0}^{N-1} (20 \times 4^i) $$

where `N` is the number of splits -- same as `ICOSAHAEDRON-NR-DIVISIONS`.


## Citation

If you use our work, please reference our paper:

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
[2] [Katz et al.: Direct visibility of point sets. ACM Transactions on Graphics 2007](https://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf)
