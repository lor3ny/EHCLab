import subprocess

def runCommand(exeName, gmonName, txtName):
    command = f"gprof {exeName} {gmonName} > {txtName}"

    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,  
        text=True             
    )


def main():
    numIterations = 50
    compiler = "O2"
    exeName = f"knn-ver1.3a/main{compiler}.exe"
    txtName = f"knn-ver1.3a/Analysis/analysis{compiler}.txt"
    gmonName = "knn-ver1.3a/gmon.out"

    d = {}

    for i in range(numIterations):
        runCommand(exeName, gmonName, txtName)

        list1 = []

        with open(txtName, "r", encoding="utf-16") as file:
            lines = file.readlines()
            for line in lines[5:]:
                l = line.strip().split()
                if not l:
                    break
                list1.append(l)

            if i:
                for el in list1:
                    d[el[-1]] = [d[el[-1]][0] + float(el[0]), d[el[-1]][1] + float(el[2])]
            
            else:
                for el in list1:
                    d[el[-1]] = [float(el[0]), float(el[2])]
    
    for key, value in d.items():
        value[0] = round(value[0] / numIterations, 2)
        value[1] = round(value[1] / numIterations, 2)

    print(d)

main()