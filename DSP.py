from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox as mb
import math
import numpy as np
import matplotlib.pyplot as plt


def SignalSamplesAreEqual(indices,samples):
    file_name = filedialog.askopenfilename(title="Choose The Compare File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
                
    if len(expected_samples)!=len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")



def readfile(filepath):
    expected_indices = []
    expected_samples = []
    with open(filepath, 'r') as f:
        SignalType = bool(f.readline().strip())
        IsPeriodic = bool(f.readline().strip())
        N1 = int(f.readline().strip())
        line = f.readline()
        while line:
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
            line = f.readline()  # move to the next line
    return expected_indices, expected_samples

def openFile():
    filepath = filedialog.askopenfilename(title="Select File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    if filepath:
        if(operations.get() == "None"):
            indices, samples = readfile(filepath)
            plotingSignal(indices, samples , len(indices))
        else:
            return filepath

def generateSignal():
    indices = []
    samples = []
    amplitude = int(mytext1.get())
    analogfrequency = int(mytext2.get())
    samplingFrequency = int(mytext3.get())
    phaseShift = float(mytext4.get())

    if(samplingFrequency < (2 *analogfrequency)  ):
        mb.showerror(title = "Sampling Theory", message = "Wrong Sampling Frequency\nhint : Sampling Frequency must be greater than 2 * Frequency")
        return

    digitalfrequency = analogfrequency / samplingFrequency

    if (mycombo1.get() == "Sin"):
        for i in range(0 ,samplingFrequency):
            firstValue = 2 * math.pi * digitalfrequency * i
            insideSin = firstValue + phaseShift
            sinValue = math.sin(insideSin)
            pointAmplitude = amplitude * sinValue

            indices.append(i)
            samples.append(round(pointAmplitude, 3))

    elif (mycombo1.get() == "Cos"):
        for i in range(0 ,samplingFrequency):
            firstValue = 2 * math.pi * digitalfrequency * i
            insideSin = firstValue + phaseShift
            sinValue = math.cos(insideSin)
            pointAmplitude = amplitude * sinValue

            indices.append(i)
            samples.append(round(pointAmplitude, 3))
    else :
        mb.showerror(title = "Signal Type Error", message = "Select A Signal Type (Sin or Cos)")
        return
    plotingSignal(indices ,samples , samplingFrequency)

def readFileDirct():
    if(operations.get() == "None"):
        openFile()
    else :
        mathOperation()

def mathOperation ():
    if(operations.get() == "Add"):
        filepath1 = openFile()
        signal1_indices , signal1_samples = readfile(filepath1)
        filepath2 = openFile()
        signal2_indices , signal2_samples = readfile(filepath2)

        if(len(signal1_indices) == len(signal2_indices)):
            arr1_samples = np.array(signal1_samples)
            arr2_samples = np.array(signal2_samples)

            arr_Of_Samples = arr1_samples + arr2_samples

            indices = signal1_indices
            samples = arr_Of_Samples.tolist()
            plotingSignal(indices,samples, len(indices) )

    elif(operations.get() == "Subtract"):
        filepath1 = openFile()
        signal1_indices , signal1_samples = readfile(filepath1)
        filepath2 = openFile()
        signal2_indices , signal2_samples = readfile(filepath2)

        if(len(signal1_indices) == len(signal2_indices)):
            arr1_samples = np.array(signal1_samples)
            arr2_samples = np.array(signal2_samples)

            arr_Of_Samples = abs(arr1_samples - arr2_samples)

            indices = signal1_indices
            samples = arr_Of_Samples.tolist()
            plotingSignal(indices,samples, len(indices) )

    elif(operations.get() == "Multiply"):
        filepath1 = openFile()
        signal1_indices , signal1_samples = readfile(filepath1)

        mutlipleVarabile = int(multi.get())
        arr1_samples = np.array(signal1_samples)
        arr1_samples *= mutlipleVarabile


        indices = signal1_indices
        samples = arr1_samples.tolist()
        plotingSignal(indices,samples, len(indices) )

    elif(operations.get() == "Square"):
        filepath1 = openFile()
        signal1_indices , signal1_samples = readfile(filepath1)

        arr1_samples = np.array(signal1_samples)

        arr_Of_Samples = pow(arr1_samples ,2)

        indices = signal1_indices
        samples = arr_Of_Samples.tolist()
        plotingSignal(indices,samples, len(indices) )

    elif(operations.get() == "Accumulate"): 
        filepath1 = openFile()
        signal1_indices , signal1_samples = readfile(filepath1)

        signal1_samples = []
        temp = 0
        for n in range(len(signal1_indices)):
            temp += n
            signal1_samples.append(temp)

        plotingSignal(signal1_indices,signal1_samples, len(signal1_indices) )


def plotingSignal(indices, samples, samplingFrequency):
    indicesArr = np.array(indices)
    samplesArr = np.array(samples)

    fig1, ax1 = plt.subplots() 
    ax1.stem(indicesArr, samplesArr) 
    ax1.set_xlim(0, samplingFrequency * 0.1)  
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Digital Signal")

    timeArr = indicesArr / samplingFrequency
    fig2, ax2 = plt.subplots()
    ax2.plot(timeArr, samplesArr) 
    ax2.set_xlim(0, 0.1) 
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Analog Signal")
    ax2.axhline(0, color='black', linewidth=1)

    plt.show()

    SignalSamplesAreEqual(indices , samples)


myframe = Tk()
myframe.title("DSP")
myframe.geometry("600x600")

# p1 = PhotoImage(file="C:/DSP/icon.png")
# myframe.iconphoto(False, p1)

mylabel =ttk.Label(myframe, text="Select Signal", font="Calibre 20 bold")
mylabel.place(relx=0.5, rely=0.5, y=-250, anchor="center")

mycombo1 = ttk.Combobox(myframe, values=["Sin", "Cos"], width=47)
mycombo1.place(relx=0.5, rely=0.5, y=-200, anchor="center")
mycombo1.current(0)

operations = ttk.Combobox(myframe, values=["None", "Add" , "Subtract" , "Multiply" , "Square" , "Normarlize" , "Accumulate"], width=47)
operations.place(relx=0.5, rely=0.5, y=-150, anchor="center")
operations.current(0)

mylab = ttk.Label(myframe, text="Amplitude")
mylab.place(relx=0.5, rely=0.5, y=-105, anchor="center")

mytext1 = ttk.Entry(myframe, width=50)
mytext1.place(relx=0.5, rely=0.5, y=-80, anchor="center")

mylab = ttk.Label(myframe, text="frequency in radians per second")
mylab.place(relx=0.5, rely=0.5, y=-45, anchor="center")

mytext2 = ttk.Entry(myframe, width=50)
mytext2.place(relx=0.5, rely=0.5,y=-20, anchor="center")

mylab = ttk.Label(myframe, text="sampling frequency")
mylab.place(relx=0.5, rely=0.5, y=15, anchor="center")

mytext3 = ttk.Entry(myframe, width=50)
mytext3.place(relx=0.5, rely=0.5, y=40, anchor="center")

mylab = ttk.Label(myframe, text="phase in radians")
mylab.place(relx=0.5, rely=0.5, y=75, anchor="center")

mytext4 = ttk.Entry(myframe, width=50)
mytext4.place(relx=0.5, rely=0.5, y=100, anchor="center")

multilab = ttk.Label(myframe, text="Multiple Value")
multilab.place(relx=0.5, rely=0.5, y=125, anchor="center")

multi = ttk.Entry(myframe, width=50)
multi.place(relx=0.5, rely=0.5, y=160, anchor="center")

mybottom =ttk.Button(myframe,text="Display Result",width=50 , command=generateSignal)
mybottom.place(relx=0.5,rely=0.5,y=210,anchor="center")

mybottom2 = ttk.Button(myframe, text="Read File", width=50, command=readFileDirct)
mybottom2.place(relx=0.5, rely=0.5, y=250, anchor="center")

def move_1(event):
    mytext2.focus()
mytext1.bind('<Return>', move_1)

def move_2(event):
    mytext3.focus()
mytext2.bind('<Return>', move_2)

def move_3(event):
    mytext4.focus()
mytext3.bind('<Return>', move_3)

def move_4(event):
    mybottom2.invoke()
mytext4.bind('<Return>', move_4)

myframe.mainloop()