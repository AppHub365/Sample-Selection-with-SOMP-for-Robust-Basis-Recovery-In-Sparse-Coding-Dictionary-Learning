# Sample Selection with SOMP for Robust Basis Recovery In Sparse Coding Dictionary Learning

## Short Description
This approach selects the samples with the highest residual error for each iteration and in theory learns distinct materials in the scene, regardless of their abundance distribution.

## Publication
DOI: [10.1109/locs.2019.2938446](https://doi.org/10.1109/locs.2019.2938446)

## Paper Abstract
Sparse Coding Dictionary (SCD) learning is to decompose a given hyperspectral image into a linear combination of a few bases. In a natural scene, because there is an imbalance in the abundance of materials, the problem of learning a given material well is directly proportional to its abundance in the training scene. By a random selection of pixels to train a given dictionary, the probability of bases learning a given material is proportional to its distribution in the scene. We propose to use SOMP residue for sample selection with each iteration for a more robust or 'more complete' learning. Experiments show that the proposed method learns from both background and trace materials accurately with over 0.95 in Pearson correlation coefficient. Furthermore, the proposed implementation has resulted in considerable improvements in Target Detection with Adaptive Cosine Estimator (ACE).
