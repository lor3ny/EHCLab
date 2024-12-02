import subprocess
import shutil
import os

def runCompiler(name):
    command = "./wincomp-gprof_"+name+".bat"
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )

def runGeneric(command):
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )

def ExecuteAndParse(d, numIterations, scenario):
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
            for line in lines[4:]:
                stats = line.strip().split()

                if not stats:
                    break
                if stats[0] == 'time':
                    start = True
                if not start:
                    continue
                
                print(stats)
                d["percentage"].append(stats[0])
                d["time"].append(stats[2])
                d["calls"].append(stats[3])
                d["name"].append(stats[len(stats)-1])
    os.remove("gmon.out")

def ImportData(scenario):
    if os.path.exists("src/data/"):
        shutil.rmtree("src/data/")
    if os.path.exists("src/params.h"):
        os.remove("src/params.h")
    runGeneric(f"cp -Recurse -Force 'scenario-{scenario}/' 'src/data'")
    runGeneric(f"cp 'scenario-{scenario}/params.h' 'src/params.h'")




def CompileAndRun(numIterations):

    if os.path.exists("benchmark_result.output"):
        os.remove("benchmark_result.output")

    with open("benchmark_result.output", "w", encoding="utf-8") as output:

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
        ExecuteAndParse(d, numIterations, "k3")
        print("Done!")
        print("Produing graphs... ")
        # graphs here
        print("Done!\n")
        
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
        ExecuteAndParse(d, numIterations, "k20")
        print("Done!")
        print("Produing graphs... ")
        # graphs here
        print("Done!\n")

    

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
        ExecuteAndParse(d, numIterations, "g100x8x5000")
        print("Done!")
        print("Produing graphs... ")
        # graphs here
        print("Done!\n")

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
        ExecuteAndParse(d, numIterations, "g100x8x1000")
        print("Done!")
        print("Produing graphs... ")
        # graphs here
        print("Done!\n")

if __name__ == "__main__":
    CompileAndRun(10)