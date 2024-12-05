# Performance Analysis and Profiling of KNN Algorithm

As part of the project for the "Efficient Heterogeneous Computing" course, we have conducted an in-depth analysis of the performance of the k-Nearest Neighbors (KNN) algorithm. Our aim was to identify bottlenecks and critical functions, visualize function usage, and assess computational time for optimizing the code on both PC/desktop and embedded systems.

## Objective
The goal was to analyze the KNN program provided, focusing on:
- Profiling via tools such as `gprof`, `gcov`, Intel VTune, and Intel Advisor.
- Identification of hotspots, call graphs, and task graphs.
- Pinpointing code regions suitable for hardware acceleration.

## Methodology
1. **Automated Script Development**:
   To streamline the profiling process, we developed a Python script that:
   - Compiles the program for various configurations.
   - Automates data collection during execution.
   - Produces detailed plots highlighting the usage frequency and execution time of different functions in the algorithm.

2. **Profiling**:
   Using tools like `gprof` and Intel VTune, we identified:
   - Critical "hotspots" in the function `knn_classifyinstance`, which dominated execution time.
   - The functions responsible for the majority of memory operations, which are potential candidates for optimization.
   - Call graphs representing the relationship and hierarchy among functions.
   - Task graphs decorated with timing and computational load for optimization analysis.

3. **Data Analysis**:
   - Collected profiling data for various scenarios (e.g., A1a, A1b, A2) as defined in the project requirements.
   - Generated visualizations (bar charts and pie charts) to compare function utilization and time distribution.

4. **Optimization Insights**:
   The profiling data revealed that:
   - The Euclidean distance calculation and sorting mechanisms are the most computationally expensive operations.
   - Parallelization of the nearest neighbor search can significantly reduce execution time.
   - Code regions with frequent calls but low computational load could be offloaded to lightweight hardware accelerators.

## Results
- **Visualization**: Graphs illustrated the time spent in each function, identifying the most "used" and the most "time-consuming" ones. These insights are pivotal in deciding the functions to optimize or migrate to hardware accelerators.
- **Hotspot Analysis**: `knn_classifyinstance` and its auxiliary functions accounted for over 80% of the execution time in large-scale scenarios.
- **Task Graph**: Highlighted computational dependencies and regions where task parallelism could be exploited.

## Conclusion
This phase of the project laid the groundwork for targeted optimizations. The automated script ensures reproducibility and flexibility in analyzing various configurations. The profiling results will guide our subsequent steps, including:
- Implementing OpenMP for multicore parallelism.
- Optimizing memory usage patterns.
- Offloading intensive computations to hardware accelerators, such as FPGAs.

## Future Work
- Evaluate the Roofline model for optimized code versions.
- Explore additional compiler optimization flags and their impact on execution time and energy consumption.
- Implement and evaluate the proposed optimizations for desktop and embedded environments.

This structured approach ensures a systematic improvement in the KNN algorithm's efficiency while maintaining scalability and adaptability for heterogeneous computing systems.

