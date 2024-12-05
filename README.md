## 1. Performance Analysis and Profiling of KNN Algorithm

As part of the project for the "Efficient Heterogeneous Computing" course, we have conducted an in-depth analysis of the performance of the k-Nearest Neighbors (KNN) algorithm. Our aim was to identify bottlenecks and critical functions, visualize function usage, and assess computational time for optimizing the code.

The analysis has been conducted on Windows 11. 
with AMD Ryzen AI 9 HX 370 and 32GB of RAM. 

**CPU Setting:**

- 12 Cores, 24 Logical processors
- From 2.0 GHz to 3.3 Ghz (if boosted 5.1 GHz)
- L1 Cache: 960 KB
- L2 Cache: 12.0 MB
- L3 Cache: 24.0 MB

### Objective
The goal was to analyze the KNN program provided, focusing on:
- Profiling via tools such as `gprof`, `gcov`, Intel VTune, and Intel Advisor.
- Identification of hotspots, call graphs.
- Pinpointing code regions suitable for hardware acceleration.

### Methodology
**Automated Script Development**: To streamline the profiling process, we developed a Python script that:
   - Compiles the program for various configurations.
   - Automates data collection during execution.
   - Produces detailed plots highlighting the usage frequency and execution time of different functions in the algorithm.

### Profiling

#### General Overview
Analyzing the plots produced we want to underline some problems:
   - Critical hotspot of the algorithm is in the function `get_knn`, which dominated execution time.
   - When K grows also `select_k_nearest` can become an hostpot
   - Using o3 flag the bottleneck remains, it is a symptom that is needed further investigation to apply parallel paradigms

#### Problems in the function `select_k_nearest` and `get_knn`:
We further analyzed the code of the functions that are responsible of more than 80% of the computation latency, these are the insights that are guiding our next optimization and acceleration offloading.

The function that starts the K-NN algorithm is called by the main function is `knn_classifyinstance`, this function is not so computationally intensive because depends directly from the function: `get_knn`.
The latter can be divided in two sections: 
1. Computing of the euclidean distance between points.
2. Select K nearest, performed calling another function.

The distance computation seems to be the hotspot because it is composed by a double loop that iterates over all the points and all the features.
This section can be parallelized using OpenMP and accelerators.

The second section become important only K grows, so it is important to take it in consideration but it is not primary hotspot.

One setting could be the implementation of a multicore paradigm on the select_k_nearest and a GPU/FPGA acceleration on the first section.
   

### Future Work
- Evaluate the Roofline model for optimized code versions.
- Implement parallelism with OpenMP, and offload intensive tasks to GPU and FPGA.
- Try on a supercomputing node.

This structured approach ensures a systematic improvement in the KNN algorithm's efficiency while maintaining scalability and adaptability for heterogeneous computing systems.

