# Sc-GNMMF
This is the code given in the "experiment" part of the paper "Graph-Regularized Non-Negative Matrix Factorization for Single-Cell Clustering in scRNA-seq Data" :
Hanjing Jiang, Mei-Neng Wang, Yu-An Huang, and Yabing Huang*. "Graph-Regularized Non-Negative Matrix Factorization for Single-Cell Clustering in scRNA-seq Data".

The code is written in the MATLAB programming language. To use, please download the SC-GNMMF folder and follow the instructions provided in the "README.doc".
Files:< br >
Data.m:It includes the matrix after cell-gene preconditioning, the cell laplacian kernel similarity matrix, and the gene laplacian kernel similarity matrix.< br >
Graph.m： Corresponding to the implementation of section "Refine Cell-Cell and Gene-Gene Similarities using p-NNG" in the paper. The input is Data.m and the output is sparse gene similarity matrix and cell similarity matrix.< br >
WKNN.m Corresponding to the implementation of section "Weighted p-NKN algorithm" in the paper.The output is the filled matrix< br >
NNDSVD.m:This is a non-negative matrix decomposition algorithm that needs to be called in main.m< br >
main.m:Corresponding to the implementation of Chapter D and Chapter E in the paper.< br >
We give the sample Data "Data.m" to run the code in its entirety.
Contact:

Please send any questions or found bugs to ybhuangwhu@163.com


