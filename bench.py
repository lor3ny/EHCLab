import subprocess
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def DrawBarPlot(data, scenario, lat):
    f, ax1 = plt.subplots(figsize=(25, 10))
    df = pd.DataFrame(data)
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale = 3)
    sns.set_palette("pastel")
    fig = sns.barplot(df, x='name', y='time', hue='name', edgecolor=".3", linewidth=.5, errorbar="sd", ax=ax1)
    i = 0
    for p in fig.patches:
        if p.get_height() == 0.0:
            i+=1
            continue
        fig.annotate(f'{p.get_height():.2f}s', (p.get_x() + p.get_width() / 2., p.get_height()+4),
                    ha='center', va='center', fontsize=25, color='red', xytext=(0, 10),
                    textcoords='offset points')
    ax1.axhline(y=lat, color='red', linestyle='--', linewidth=2, label=f'Average Total Latency: {1.243}')

    ax1.set_xlabel('')

    #plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.ylabel('Latency (s)', fontsize=28);
    plt.legend()
    #plt.title(f'Optimization and Accelerators Latencies Scenario {scenario}', fontsize=28)
    plt.tight_layout(h_pad=2)

    plt.savefig(f'plot_latency_{scenario}.png')

def DrawBarPlot(data, scenario, lat):
    f, ax1 = plt.subplots(figsize=(25, 10))
    df = pd.DataFrame(data)
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale = 2.5)
    sns.set_palette("pastel")
    fig = sns.barplot(df, x='name', y='time', hue='name', edgecolor=".3", linewidth=.5, errorbar="sd", ax=ax1)
    i = 0
    for p in fig.patches:
        if data['spups'][i] == 1:
            i+=1
            continue
        print(data['spups'][i])
        if data['spups'][i] < 1:
            fig.annotate(f'{data['spups'][i]:.2f} Speedup :(', (p.get_x() + p.get_width() / 2., p.get_height()+0.03),
                    ha='center', va='center', fontsize=30, color='red', xytext=(0, 10),
                    textcoords='offset points')
        else:
            fig.annotate(f'{data['spups'][i]:.2f} Speedup :)', (p.get_x() + p.get_width() / 2., p.get_height()+0.03),
                    ha='center', va='center', fontsize=30, color='red', xytext=(0, 10),
                    textcoords='offset points')
        i+=1
    ax1.axhline(y=1.243, color='red', linestyle='--', linewidth=2, label=f'Default Total Latency: {1.243}s')

    ax1.axhline(y=1.222, color='green', linestyle='--', linewidth=2, label=f'FPGA Optimized Total Latency: {1.223}s')

    ax1.axhline(y=1.951, color='blue', linestyle='--', linewidth=2, label=f'FPGA Not Optimized Total Latency: {1.951}s')

    ax1.set_xlabel('')

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.ylabel('Latency (s)', fontsize=30)
    plt.legend()
    #plt.title(f'Optimization and Accelerators Latencies Scenario {scenario}', fontsize=28)
    plt.tight_layout(h_pad=2)

    plt.savefig(f'plot_latency_{scenario}.png')

def DrawSpeedBarPlot(data, scenario, lat):
    f, ax1 = plt.subplots(figsize=(25, 10))
    df = pd.DataFrame(data)
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale = 3)
    sns.set_palette("pastel")

    fig = sns.barplot(df, x='name', y='spups', hue='type' , edgecolor=".3", linewidth=.5, errorbar="sd", ax=ax1)

    ax1.set_xlabel('')

    ax1.axhline(y=lat, color='red', linestyle='--', linewidth=2, label=f'Average Total Latency: {lat}')

    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=18.5)
    plt.ylabel('Latency (s)', fontsize=28)
    #plt.title('Optimization and Accelerators Speedups Scenario {scenario}', fontsize=28)
    plt.legend()
    plt.tight_layout(h_pad=2)

    plt.savefig(f'plot_spups_{scenario}.png')

def runGeneric(command):
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )
    print(result.stderr)


def ParseLatency(data, spups, name, default_times):
    values = []
    with open("C:/Dev/EHCLab/knn_cuda/build/profile.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            start = False

            min_def = min(default_times)
            max_def = max(default_times)

            prev_line = None
            for line in lines:
                stats = line.strip().split()
                value = float(stats[0])
                values.append(value)
                data['time'].append(value)
                data['type'].append(name)

    spups['spups'].append(min_def/max(values))
    spups['spups'].append(max_def/min(values))
    spups['name'].append(name)
    spups['name'].append(name)
    spups['type'].append("min speedup")
    spups['type'].append("max speedup")
                


    # Build the dictionary

def CreateFile():
    file_name = "C:/Dev/EHCLab/knn_cuda/build/profile.txt"

    # Open the file in write mode
    with open(file_name, "w") as file:
        pass 


if __name__ == "__main__":


    '''
    try:
        os.remove("build/profile.txt")
    except:
        print("Profile.txt don't exist")
    CreateFile()
    for i in range(15):
        print(i)
        runGeneric("./build/knn_default.exe")
    '''

    runs = {
            "time": [],
            "type": [],
        }

    spups = {
        'spups': [],
        'name': [],
        'type': []
    }
    
    default_times = [268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.838020, 270.107559, 237.523171]

    '''
    # cuda heap v1
    try:
        os.remove("build/profile.txt")
    except:
        print("Profile.txt don't exist")
    CreateFile()
    for i in range(10):
        print(i)
        runGeneric("./build/knn_cuda_heap_v1.exe")

    ParseLatency(runs, spups, "CUDA v1 - Min-Heap", default_times)
    os.remove("C:/Dev/EHCLab/knn_cuda/build/profile.txt")
    print(runs)
    print(spups)

    
    # cuda heap v2
    for i in range(10):
        print(i)
        runGeneric("./build/knn_cuda_heap_v2.exe")

    ParseLatency(runs, spups, "CUDA v2 - Min-Heap", default_times)
    os.remove("build/profile.txt")
    print(runs)
    print(spups)

    # cuda quicksort
    for i in range(10):
        print(i)
        runGeneric("./build/knn_cuda_quicksort_v1.exe")

    ParseLatency(runs, spups, "CUDA v1 - Quick Sort", default_times)
    os.remove("build/profile.txt")
    print(runs)
    print(spups)

    # cuda quicksort
    for i in range(10):
        print(i)
        runGeneric("./build/knn_cuda_quicksort_v2.exe")

    ParseLatency(runs, spups, "CUDA v2 - Quick Sort", default_times)
    os.remove("build/profile.txt")
    print(runs)
    print(spups)

    # omp heap
    for i in range(10):
        print(i)
        runGeneric("./build/knn_omp_heap.exe")

    ParseLatency(runs, spups, "Open MP - Min-Heap", default_times)
    os.remove("build/profile.txt")
    print(runs)
    print(spups)

    # omp quicksort
    for i in range(10):
        print(i)
        runGeneric("./build/knn_omp_quicksort.exe")

    ParseLatency(runs, spups, "Open MP - Quick Sort", default_times)
    os.remove("build/profile.txt")
    print(runs)
    print(spups)
    '''


    '''
    runs = {
        'time': [26.990333, 26.259858, 26.330355, 26.461906, 26.413402, 26.35447, 25.894687, 26.072652, 26.85155, 26.503415, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171, 31.775372, 32.060705, 32.193076, 32.047718, 32.133023, 32.075081, 31.798549, 32.005748, 31.936399, 31.750092, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171, 29.216373, 28.658915, 28.816417, 28.656658, 28.670217, 28.847525, 29.05877, 28.98637, 29.004514, 28.968269, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171, 34.734001, 34.51681, 34.368496, 34.335135, 34.2612, 33.857526, 33.956587, 33.523692, 33.56921, 33.561167, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171, 6.676288, 6.409632, 6.353498, 6.456091, 7.782443, 7.906929, 7.147424, 6.915611, 6.653417, 6.994499, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171, 8.732794, 8.616473, 9.172056, 9.116622, 8.927718, 8.753403, 8.721827, 8.770015, 8.794486, 8.724795, 268.184519, 282.549767, 264.607876, 270.890002, 316.915238, 310.464082, 286.712508, 327.83802, 270.107559, 237.523171], 
        'type': ['CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default', 'Default'], 
        'group': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F']
    }
    '''

    runs = {
        'time': [28.174404, 26.061803, 26.435191, 26.25397, 27.519745, 26.354714, 26.363819, 26.296124, 26.399245, 26.787277, 31.951144, 32.045728, 32.008145, 32.036677, 31.877051, 31.722595, 31.814887, 31.929403, 32.068334, 32.167813, 29.06561, 29.128415, 29.179368, 28.947576, 28.768507, 28.817171, 29.04534, 28.833383, 28.672489, 28.738605, 34.345162, 34.258131, 34.273609, 36.924068, 34.732413, 34.551322, 34.810253, 34.66371, 34.626881, 34.498001, 8.274905, 6.732815, 6.733444, 6.527141, 6.480696, 6.380442, 6.398697, 6.410545, 6.351079, 6.868183, 8.581318, 8.603074, 8.484997, 8.529824, 8.480258, 8.556841, 8.466726, 8.472908, 8.462152, 8.469124], 
        'type': ['CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort', 'Open MP - Quick Sort']
    }

    spups = {
        'spups': [8.430459469524182, 12.579253246600013, 7.38387688961012, 10.334527172193825, 8.14010676996157, 11.433887724222338, 6.432746548944715, 9.569641145922409, 28.70403599799635, 51.619263435394195, 27.609104722335296, 38.74168414842938], 
        'name': ['CUDA v1 - Min-Heap', 'CUDA v1 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v2 - Min-Heap', 'CUDA v1 - Quick Sort', 'CUDA v1 - Quick Sort', 'CUDA v2 - Quick Sort', 'CUDA v2 - Quick Sort', 'Open MP - Min-Heap', 'Open MP - Min-Heap', 'Open MP - Quick Sort', 'Open MP - Quick Sort'], 'type': ['min speedup', 'max speedup', 'min speedup', 'max speedup', 'min speedup', 'max speedup', 'min speedup', 'max speedup', 'min speedup', 'max speedup', 'min speedup', 'max speedup']
    }

    default_runs = {
        'spups': [203.95, 204.98, 200.12, 0.7545, 0.6543, 0.7345, 20.45, 12.12, 16.12, 0.122, 0.098, 0.132, 0, 0, 0, 0.012, 0.0089, 0.018, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'name': ['get_k_NN', 'get_k_NN', 'get_k_NN','get_k_NN', 'get_k_NN', 'get_k_NN', 'select_k_nearest', 'select_k_nearest', 'select_k_nearest', 'select_k_nearest', 'select_k_nearest', 'select_k_nearest', 'minmax', 'minmax', 'minmax', 'minmax', 'minmax', 'minmax','minmax_normalize_point', 'minmax_normalize_point', 'minmax_normalize_point', 'minmax_normalize_point', 'minmax_normalize_point', 'minmax_normalize_point', 'minmax_normalize', 'minmax_normalize', 'minmax_normalize', 'minmax_normalize', 'minmax_normalize', 'minmax_normalize', 'main', 'main', 'main', 'main', 'main', 'main', 'copy_k_nearest', 'copy_k_nearest', 'copy_k_nearest', 'copy_k_nearest', 'copy_k_nearest', 'copy_k_nearest', 'knn_classifyinstance', 'knn_classifyinstance', 'knn_classifyinstance', 'knn_classifyinstance', 'knn_classifyinstance', 'knn_classifyinstance', 'plurality_voting', 'plurality_voting', 'plurality_voting', 'plurality_voting', 'plurality_voting', 'plurality_voting'],
        'type': ['g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM',  'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM', 'g100x8x50000', 'g100x8x50000', 'g100x8x50000', 'WISDM', 'WISDM', 'WISDM']
    }

    fpga = {
        'time': [0.234, 0.213, 0.941],
        'spups': [1 , 1.09, 0.24],
        'name': ['CPU Sequential', 'FPGA Optimized', 'FPGA Not Optimized']
    }


    wei_lat = 0
    default_times
    for i in range(10):
        wei_lat += default_times[i]
    wei_lat /= 10

    DrawBarPlot(fpga, "g100x8x50000", 2.5)
    #DrawSpeedBarPlot(default_runs, "g100x8x50000", 222.089)

    #DrawBarPlot(runs, "g100x8x50000", wei_lat)
    #DrawSpeedBarPlot(spups, "g100x8x50000")