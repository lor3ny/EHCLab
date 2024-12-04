import subprocess
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def runCompiler(name):
    command = "./wincomp-gprof_"+name+".bat"
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )
    print(result)

def runGeneric(command):
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )

def ExecuteAndParse(d, numIterations, scenario):


    times = []
    for i in range(numIterations+1):

        if i == 0:
            print(f"{i}/{numIterations} WARM UP")
        else:
            print(f"{i}/{numIterations} ")

        #profCommand = ".\\build\knn.exe"
        #runGeneric(profCommand)
        #profCommand = f"gprof .\\build\knn.exe > .\profiling\profile{scenario}.outputs"
        #runGeneric(profCommand)
        
        runGeneric("./profile.bat")

        if i == 0:
            continue

        with open("profiling/profile.output", "r", encoding="utf-8") as file:
            lines = file.readlines()
            start = False

            prev_line = None
            for line in lines[4:]:
                stats = line.strip().split()

                if not stats:
                    os.remove("gmon.out")
                    times.append(float(prev_line[1]))
                    break
                
                if stats[0] == 'time':
                    start = True
                    continue
                if not start:
                    continue
                
                try:
                    d["percentage"].append(float(stats[0]))
                except ValueError:
                    d["percentage"].append(0)
                try:
                    d["time"].append(float(stats[2]))
                except ValueError:
                    d["time"].append(0)
                try:
                    d["calls"].append(float(stats[3]))
                except ValueError:
                    d["calls"].append(0)
                try:
                    d["name"].append(stats[len(stats)-1])
                except ValueError:
                    d["name"].append("null")
                prev_line = stats
    return times

def ImportData(scenario):
    if os.path.exists("src/data/"):
        shutil.rmtree("src/data/")
    if os.path.exists("src/params.h"):
        os.remove("src/params.h")
    runGeneric(f"cp -Recurse -Force 'scenario-{scenario}/' 'src/data'")
    runGeneric(f"cp 'scenario-{scenario}/params.h' 'src/params.h'")


def DrawGraph(data, scenario, lat):

    f, ax1 = plt.subplots(figsize=(25, 10))
    df = pd.DataFrame(data)
    sns.set_theme(style="white", context="paper")
    sns.barplot(df,x='name', y='time', hue='name', palette="light:m_r", edgecolor=".3", linewidth=.5, ax=ax1)
    
    ax1.axhline(y=lat, color='red', linestyle='--', linewidth=1.5, label=f'Average Total Latency: {lat}')

    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel('Latency (s)')
    ax1.set_xlabel(f'Functions on {scenario} K-NN')
    plt.legend()
    plt.title('Function Latencies')
    plt.tight_layout(h_pad=2)

    plt.savefig(f'profiling/graphs/plot_latency_{scenario}.png')

    f, ax2 = plt.subplots(figsize=(25, 10))
    sns.set_theme(style="white", context="paper")
    sns.barplot(df,x='name', y='calls', hue='name', palette="light:m_r", edgecolor=".3", linewidth=.5, ax=ax2)
    
    ax2.set_ylabel('Calls Count')
    ax2.set_xlabel(f'Functions on {scenario} K-NN')
    plt.legend()
    plt.title('Function Calls')
    plt.tight_layout(h_pad=2)

    plt.savefig(f'profiling/graphs/plot_calls_{scenario}.png')

def DrawGraphFinal(data):

    f, ax1 = plt.subplots(figsize=(20, 10))
    df = pd.DataFrame(data)
    sns.set_theme(style="white", context="talk")
    sns.barplot(df,x='scenario', y='time', hue='scenario', palette="light:m_r", edgecolor=".3", linewidth=.5, ax=ax1)
    
    #ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel('Latency (s)')
    ax1.set_xlabel(f'All Scenarios K-NN')
    plt.title('Function Latencies')
    plt.tight_layout(h_pad=2)

    plt.savefig(f'profiling/graphs/plot_total.png')
    return



def CompileAndRun(numIterations):

    if os.path.exists("benchmark_result.output"):
        os.remove("benchmark_result.output")

    with open("benchmark_result.output", "w", encoding="utf-8") as output:

        runs = {
            "time": [],
            "scenario": []
        }

        '''
        print("SIMPLE SCENARIO")
        print("Import data... ")
        ImportData("simple")
        print("Done!")
        print("Compiling... ")
        runCompiler("simple")
        print("Done!")
        d = {
            "percentage": [],
            "time": [],
            "calls": [],
            "name": []
        }
        print("Executing and Parsing... ")
        ExecuteAndParse(d, numIterations, "simple")
        print("Done!")
        print("Produing graphs... ")
        # graphs here
        print("Done!\n")
        '''
        

        print("K3 SCENARIO")
        print("Import data... ")
        ImportData("wisdm")
        print("Done!")
        print("Compiling... ")
        runCompiler("k3")
        print("Done!")
        d = {
            "percentage": [],
            "time": [],
            "calls": [],
            "name": []
        }
        print("Executing and Parsing... ")
        total_time = ExecuteAndParse(d, numIterations, "k3")
        print("Done!")
        print("Produing graphs... ")
        DrawGraph(d, "k3", sum(total_time)/len(total_time))
        print("Done!\n")

        for i in range(len(total_time)):
            runs["scenario"].append('k3')
            runs["time"].append(total_time[i])
        
        print("K20 SCENARIO")
        print("Import data... ")
        ImportData("wisdm")
        print("Done!")
        print("Compiling... ")
        runCompiler("k20")
        print("Done!")
        d = {
            "percentage": [],
            "time": [],
            "calls": [],
            "name": []
        }
        print("Executing and Parsing... ")
        total_time = ExecuteAndParse(d, numIterations, "k20")
        print("Done!")
        print("Produing graphs... ")
        DrawGraph(d, "k20", sum(total_time)/len(total_time))
        print("Done!\n")

        for i in range(len(total_time)):
            runs["scenario"].append('k20')
            runs["time"].append(total_time[i])

        print("g100x8x5000 SCENARIO")
        print("Import data... ")
        ImportData("g100x8x5000")
        print("Done!")
        print("Compiling... ")
        runCompiler("g100x8x5000")
        print("Done!")
        d = {
            "percentage": [],
            "time": [],
            "calls": [],
            "name": []
        }
        print("Executing and Parsing... ")
        total_time = ExecuteAndParse(d, numIterations, "g100x8x5000")
        print("Done!")
        print("Produing graphs... ")
        DrawGraph(d, "g100x8x5000", sum(total_time)/len(total_time))
        print("Done!\n")

        for i in range(len(total_time)):
            runs["scenario"].append('g100x8x5000')
            runs["time"].append(total_time[i])

        print("g100x8x1000 SCENARIO")
        print("Import data... ")
        ImportData("g100x8x1000")
        print("Done!")
        print("Compiling... ")
        runCompiler("g100x8x1000")
        print("Done!")
        d = {
            "percentage": [],
            "time": [],
            "calls": [],
            "name": []
        }
        print("Executing and Parsing... ")
        total_time = ExecuteAndParse(d, numIterations, "g100x8x1000")
        print("Done!")
        print("Produing graphs... ")
        DrawGraph(d, "g100x8x1000", sum(total_time)/len(total_time))
        print("Done!\n")

        for i in range(len(total_time)):
            runs["scenario"].append('g100x8x1000')
            runs["time"].append(total_time[i])

        DrawGraphFinal(runs)

if __name__ == "__main__":
    CompileAndRun(10)