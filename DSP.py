from tkinter import *
from tkinter import ttk, Toplevel, Label, Entry, Button, StringVar
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox as mb
from PIL import Image, ImageTk
import math
import numpy as np
import matplotlib.pyplot as plt

def get_two_inputs():
    choice = simpledialog.askstring("Choice", "Do you want to enter num_bits or num_levels? (Type 'bits' or 'levels')")

    if choice == 'bits':
        num_bits = simpledialog.askstring("Input", "Enter num_bits:")
        num_levels = 2 ** int(num_bits)
    elif choice == 'levels':
        num_levels = simpledialog.askstring("Input", "Enter num_levels:")
        num_bits = int(math.log2(int(num_levels)))
    else:
        mb.showerror("Input Error", "Please type 'bits' or 'levels'.")
        return None, None

    return num_bits, num_levels

def SignalSamplesAreEqual(indices,samples):
    file_name = filedialog.askopenfilename(title="Choose The Compare File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:

            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break

    if len(expected_samples) != len(samples):
        mb.showerror("Test Case Failed", "Your signal has a different length from the expected one.")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            mb.showerror("Test Case Failed", "Your signal has different values from the expected one.")
            return
    mb.showinfo("Test Case Passed", "Test case passed successfully.")

def QuantizationTest1(Your_EncodedValues,Your_QuantizedValues):
    file_name = filedialog.askopenfilename(title="Choose The Compare File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
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
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        mb.showerror("Test Case Failed", "Your signal has a different length from the expected one.")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            mb.showerror("Test Case Failed", "Your  EncodedValues have different EncodedValues from the expected one.")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            mb.showerror("Test Case Failed", "your QuantizedValues have different values from the expected one.")
            return
    mb.showinfo("Test Case Passed", "QuantizationTest1 Test case passed successfully.")

def QuantizationTest2(Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
    file_name = filedialog.askopenfilename(title="Choose The Compare File", filetypes=(("text files", ".txt"), ("all files", ".*")))
    expectedIntervalIndices=[]
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    expectedSampledError=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==4:
                L=line.split(' ')
                V1=int(L[0])
                V2=str(L[1])
                V3=float(L[2])
                V4=float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
     or len(Your_EncodedValues)!=len(expectedEncodedValues)
      or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
      or len(Your_SampledError)!=len(expectedSampledError)):
        mb.showerror("Test Case Failed", "Your signal has a different length from the expected one.")
        return
    for i in range(len(Your_IntervalIndices)):
        if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
            mb.showerror("Test Case Failed", "Your signal has a different indicies from the expected one.")
            return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            mb.showerror("Test Case Failed", "Your EncodedValues have different EncodedValues from the expected one.")
            return
        
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            mb.showerror("Test Case Failed", "your QuantizedValues have different values from the expected one.")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            mb.showerror("Test Case Failed", "Your signal has a different length from the expected one.")
            return
    mb.showinfo("Test Case Passed", "QuantizationTest1 Test case passed successfully.")

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

    if samplingFrequency < (2 *analogfrequency):
        mb.showerror("Sampling Theory", "Wrong Sampling Frequency\nhint : Sampling Frequency must be greater than 2 * Frequency")
        return

    digitalfrequency = analogfrequency / samplingFrequency

    if mycombo1.get() == "Sin":
        for i in range(0, samplingFrequency):
            firstValue = 2 * math.pi * digitalfrequency * i
            insideSin = firstValue + phaseShift
            sinValue = math.sin(insideSin)
            pointAmplitude = amplitude * sinValue

            indices.append(i)
            samples.append(round(pointAmplitude, 3))

    elif mycombo1.get() == "Cos":
        for i in range(0, samplingFrequency):
            firstValue = 2 * math.pi * digitalfrequency * i
            insideSin = firstValue + phaseShift
            sinValue = math.cos(insideSin)
            pointAmplitude = amplitude * sinValue

            indices.append(i)
            samples.append(round(pointAmplitude, 3))
    else:
        mb.showerror("Signal Type Error", "Select A Signal Type (Sin or Cos)")
        return
    plotingSignal(indices, samples, samplingFrequency)

def readFileDirct():
    if operations.get() == "None":
        openFile()
    else:
        mathOperation()


def normalize_signal(signal, range_type='-1 to 1'):

    min_val = np.min(signal)
    max_val = np.max(signal)

    if range_type == '-1 to 1':

        normalized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1
    elif range_type == '0 to 1':

        normalized_signal = (signal - min_val) / (max_val - min_val)
    else:
        raise ValueError("range_type must be either '-1 to 1' or '0 to 1'.")

    return normalized_signal

def quantize_signal():
    filepath = openFile()
    num_bits, num_levels = get_two_inputs()
    num_levels = int(num_levels)
    num_bits = int(num_bits)

    indices, samples = readfile(filepath)
    signal = np.array(samples)

    min_amp = np.min(signal)
    max_amp = np.max(signal)
    delta = (max_amp - min_amp) / num_levels

    intVaral = []
    midpoints = []
    start = min_amp

    for i in range(num_levels):
        end = round(start + delta, 5)
        intVaral.append([start, end])
        midpoints.append(round((start + end) / 2, 5))
        start = end 

    quantized_signal = np.zeros_like(signal)
    interval_indices = []

    for i in range(len(signal)):
        for j in range(num_levels):
            if intVaral[j][0] < signal[i] <= intVaral[j][1]:
                quantized_signal[i] = midpoints[j]
                interval_indices.append(j+1)
                break
            elif(signal[i] == min_amp):
                quantized_signal[i] = midpoints[0]
                interval_indices.append(1)
                break

    error = quantized_signal - signal
    N = len(signal)
    average_error = np.sum(error) / N

    indices = list(indices)
    samples = list(quantized_signal)

    interval_indices_binary = []
    for i in range(len(quantized_signal)):
        bit_string = format(interval_indices[i]-1, f'0{num_bits}b')
        interval_indices_binary.append(bit_string)


    mb.showinfo("Average Error", "The average error is: " + str(average_error))

    plotingSignal(indices, samples, len(indices))

    chooseTest = mb.askquestion("Which Test", "For the first test enter yes otherwise enter no")

    if(chooseTest == "yes"):
        QuantizationTest1(interval_indices_binary, samples)
    else:
        QuantizationTest2(interval_indices, interval_indices_binary, samples, error)


def mathOperation ():

    if operations.get() == "Add":
        filepath1 = openFile()
        signal1_indices, signal1_samples = readfile(filepath1)
        filepath2 = openFile()
        signal2_indices, signal2_samples = readfile(filepath2)

        if len(signal1_indices) == len(signal2_indices):
            arr1_samples = np.array(signal1_samples)
            arr2_samples = np.array(signal2_samples)

            arr_Of_Samples = arr1_samples + arr2_samples

            indices = signal1_indices
            samples = arr_Of_Samples.tolist()
            plotingSignal(indices, samples, len(indices))

    elif operations.get() == "Subtract":
        filepath1 = openFile()
        signal1_indices, signal1_samples = readfile(filepath1)
        filepath2 = openFile()
        signal2_indices, signal2_samples = readfile(filepath2)

        if(len(signal1_indices) == len(signal2_indices)):
            arr1_samples = np.array(signal1_samples)
            arr2_samples = np.array(signal2_samples)

            arr_Of_Samples = abs(arr1_samples - arr2_samples)

            indices = signal1_indices
            samples = arr_Of_Samples.tolist()
            plotingSignal(indices, samples, len(indices))

    elif operations.get() == "Multiply":
        filepath1 = openFile()
        signal1_indices, signal1_samples = readfile(filepath1)

        mutlipleVarabile = int(multi.get())
        arr1_samples = np.array(signal1_samples)
        arr1_samples *= mutlipleVarabile


        indices = signal1_indices
        samples = arr1_samples.tolist()
        plotingSignal(indices, samples, len(indices))

    elif operations.get() == "Square":
        filepath1 = openFile()
        signal1_indices, signal1_samples = readfile(filepath1)

        arr1_samples = np.array(signal1_samples)

        arr_Of_Samples = pow(arr1_samples, 2)

        indices = signal1_indices
        samples = arr_Of_Samples.tolist()
        plotingSignal(indices, samples, len(indices))

    elif operations.get() == "Accumulate":
        filepath1 = openFile()
        signal1_indices, signal1_samples = readfile(filepath1)

        signal1_samples = []
        temp = 0
        for n in range(len(signal1_indices)):
            temp += n
            signal1_samples.append(temp)

        plotingSignal(signal1_indices, signal1_samples, len(signal1_indices))


    elif operations.get() == "Normalize":

        filepath1 = openFile()

        signal1_indices, signal1_samples = readfile(filepath1)


        range_type = mb.askquestion("Normalization", "Normalize to '0 to 1' or '-1 to 1'? (default is '0 to 1')")

        range_type = '0 to 1' if range_type == 'yes' else '-1 to 1'

        arr1_samples = np.array(signal1_samples)

        normalized_samples = normalize_signal(arr1_samples, range_type)

        indices = signal1_indices

        samples = normalized_samples.tolist()

        plotingSignal(indices, samples, len(indices))

    elif operations.get() == "Quantize":
        quantize_signal()

def plotingSignal(indices, samples, samplingFrequency):
    indicesArr = np.array(indices)
    samplesArr = np.array(samples)

    fig1, ax1 = plt.subplots() 
    ax1.stem(indicesArr, samplesArr) 
    # ax1.set_xlim(0, samplingFrequency * 0.1)  
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Digital Signal")

    timeArr = indicesArr / samplingFrequency
    fig2, ax2 = plt.subplots()
    ax2.plot(timeArr, samplesArr) 
    # ax2.set_xlim(0, 0.1) 
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Analog Signal")
    ax2.axhline(0, color='black', linewidth=1)

    plt.show()

    if(operations.get() != "Quantize"):
        SignalSamplesAreEqual(indices, samples)


myframe = Tk()
myframe.title("DSP")
myframe.geometry("1800x900")

p1 = PhotoImage(file="icon.png")
myframe.iconphoto(False, p1)

image_path = "img.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((1600, 900))
background_image = ImageTk.PhotoImage(resized_image)
background_label = ttk.Label(myframe, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

mylabel =ttk.Label(myframe, text="Select Signal", font="Calibre 20 bold")
mylabel.place(relx=0.5, rely=0.5, y=-250, anchor="center")

mycombo1 = ttk.Combobox(myframe, values=["Sin", "Cos"], width=47)
mycombo1.place(relx=0.5, rely=0.5, y=-200, anchor="center")
mycombo1.current(0)

label =ttk.Label(myframe, text="Arithmetic Operations", font="Calibre 20 bold")
label.place(relx=0.5, rely=0.5, x=400, y=-250, anchor="center")

operations = ttk.Combobox(myframe, values=["None", "Add", "Subtract", "Multiply", "Square", "Normalize", "Accumulate", "Quantize"], width=47)
operations.place(relx=0.5, rely=0.5, x=400, y=-200, anchor="center")
operations.current(0)

mylab = ttk.Label(myframe, text="Amplitude")
mylab.place(relx=0.5, rely=0.5, x=-450, y=-105, anchor="center")

mytext1 = ttk.Entry(myframe, width=50)
mytext1.place(relx=0.5, rely=0.5, x=-450, y=-80, anchor="center")

mylab = ttk.Label(myframe, text="frequency in radians per second")
mylab.place(relx=0.5, rely=0.5, x=-450, y=-45, anchor="center")

mytext2 = ttk.Entry(myframe, width=50)
mytext2.place(relx=0.5, rely=0.5, x=-450, y=-20, anchor="center")

mylab = ttk.Label(myframe, text="sampling frequency")
mylab.place(relx=0.5, rely=0.5, x=-450, y=15, anchor="center")

mytext3 = ttk.Entry(myframe, width=50)
mytext3.place(relx=0.5, rely=0.5, x=-450, y=40, anchor="center")

mylab = ttk.Label(myframe, text="phase in radians")
mylab.place(relx=0.5, rely=0.5, x=-450, y=75, anchor="center")

mytext4 = ttk.Entry(myframe, width=50)
mytext4.place(relx=0.5, rely=0.5, x=-450, y=100, anchor="center")

multilab = ttk.Label(myframe, text="Multiple Value")
multilab.place(relx=0.5, rely=0.5, x=-450, y=135, anchor="center")

multi = ttk.Entry(myframe, width=50)
multi.place(relx=0.5, rely=0.5, x=-450, y=160, anchor="center")

mybottom =ttk.Button(myframe,text="Display Result",width=50 , command=generateSignal)
mybottom.place(relx=0.5, rely=0.5, y=210,anchor="center")

mybottom2 = ttk.Button(myframe, text="Read File", width=50, command=readFileDirct)
mybottom2.place(relx=0.5, rely=0.5, x=400, y=210, anchor="center")

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