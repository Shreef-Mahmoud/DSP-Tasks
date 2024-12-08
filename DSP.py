from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox as mb
from tkinter.filedialog import asksaveasfilename
from scipy.signal import convolve
from PIL import Image, ImageTk
import math
import numpy as np
import matplotlib.pyplot as plt


def SignalComapreAmplitude(SignalInput, SignalOutput):

    if len(SignalInput) != len(SignalOutput):
        print("Error: SignalInput and SignalOutput have different lengths.")
        return False


    for i in range(len(SignalInput)):
        if abs(float(SignalInput[i]) - float(SignalOutput[i])) > 0.001:
            print(f"Amplitude mismatch at index {i}")
            return False

    return True

def RoundPhaseShift(P):

    while P<0:
        P+=2*math.pi
    return float(P%(2*math.pi))


def SignalComaprePhaseShift(SignalInput = [] ,SignalOutput= []):
    if len(SignalInput) != len(SignalInput):
        return False
    else:
        for i in range(len(SignalInput)):
            A=round(SignalInput[i])
            B=round(SignalOutput[i])
            if abs(A-B)>0.0001:
                return False
            elif A!=B:
                return False
        return True

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

def SignalAreEqual(samples):
    file_name = filedialog.askopenfilename(title="Choose The Compare File", filetypes=(("text files", ".txt"), ("all files", ".*")))
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
                V2 = float(L[1])
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

def Compare_Signals(Your_indices,Your_samples):
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
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        mb.showerror("Shift_Fold_Signal Test case failed", "your signal have different values from the expected one.")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            mb.showerror("Shift_Fold_Signal Test case failed", "your signal have different indicies from the expected one.")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            mb.showerror("Correlation Test case failed", "your signal have different values from the expected one.")
            return
    mb.showinfo("Test Case Passed", "Correlation Test case passed successfully")

def Shift_Fold_Signal(Your_indices,Your_samples):
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
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    mb.showinfo("Test Case Passed", "Shift_Fold_Signal Test case passed successfully")

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
            line = f.readline()
    return expected_indices, expected_samples

def readfile_DFT(filepath):
    expected_indices = []
    expected_samples = []

    with open(filepath, 'r') as f:
        SignalType = f.readline().strip().lower() == 'true'
        IsPeriodic = f.readline().strip().lower() == 'true'
        N1 = int(f.readline().strip())



        line = f.readline()
        while line:
            L = line.strip()


            if ',' in L:
                L = L.split(',')
            else:
                L = L.split()


            if len(L) != 2:
                line = f.readline()
                continue


            V1_str = L[0].strip()
            if V1_str.endswith('f'):
                V1_str = V1_str[:-1]
            try:
                V1 = float(V1_str)
            except ValueError:
                line = f.readline()
                continue


            V2_str = L[1].strip()
            if V2_str.endswith('f'):
                V2_str = V2_str[:-1]
            try:
                V2 = float(V2_str)
            except ValueError:
                line = f.readline()
                continue

            expected_indices.append(V1)
            expected_samples.append(V2)

            line = f.readline()

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

def DFT_Operation():
    filepath = openFile()
    indices, samples = readfile(filepath)

    DFTresult = []

    for i in range(len(indices)):
        temp = 0 + 0j
        for x in range(len(indices)):
            a = samples[x] * (math.cos(2 * math.pi * i * x / len(indices)) - math.sin(2 * math.pi * i * x / len(indices)) * 1j)
            temp += a
        DFTresult.append(temp)

    amplitude_list = []
    angle_list = []

    if operations.get() == "DC_component_freq":
        return list(indices) , list(DFTresult)

    for i in range(len(DFTresult)):
        angle = math.atan2(np.imag(DFTresult[i]),  np.real(DFTresult[i]))
        amplitude = math.sqrt(math.pow(np.real(DFTresult[i]), 2) + math.pow(np.imag(DFTresult[i]), 2))
        amplitude_list.append(amplitude)
        angle_list.append(angle)




    fs = simpledialog.askstring("Input", "Enter The Sampling Frequency")
    fs = float(fs)
    ts = 1/fs

    omega = (2 * math.pi) / (ts * len(DFTresult))

    frequency_domain_indices = []
    temp = 0
    for i in range(len(DFTresult)):
        temp += omega
        frequency_domain_indices.append(temp)

    indicesArr = np.array(frequency_domain_indices)
    samplesArr_amplitude = np.array(amplitude_list)
    samplesArr_Angle = np.array(angle_list)

    fig1, ax1 = plt.subplots()
    ax1.stem(indicesArr, samplesArr_amplitude)
    ax1.set_xlabel("Amplitude")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Signal Amplitude Frequncy Domain")


    fig1, ax1 = plt.subplots()
    ax1.stem(indicesArr, samplesArr_Angle)
    ax1.set_xlabel("Angle (in radian)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Signal Angle Frequncy Domain")

    plt.show()

    Output = openFile()
    amp, phase = readfile_DFT(Output)

    if(SignalComapreAmplitude(amplitude_list , amp)):
        mb.showinfo("Test Case Passed", "Amplitude Test case passed successfully.")

    for i in range(len(angle_list)):
        angle_list[i] = RoundPhaseShift(angle_list[i])
        angle_list[i] = round(angle_list[i], 7)
        phase[i] = RoundPhaseShift(phase[i])
        phase[i] = round(phase[i], 7)

    if(SignalComaprePhaseShift(angle_list , phase)):
        mb.showinfo("Test Case Passed", "Phase Shift Test case passed successfully.")



def IDFT_Operation( amp_para = [] , phase_para = []):
    if operations.get() == "DC_component_freq":
        amp = list(amp_para)
        phase = list(phase_para)
    else:
        filepath = openFile()
        amp, phase = readfile_DFT(filepath)

    N = len(amp)
    IDFTresult = []




    values = np.array([amplitude * (math.cos(phase) + 1j * math.sin(phase)) for amplitude, phase in zip(amp, phase)])

    for n in range(N):
        temp = 0 + 0j
        for k in range(N):
            a = values[k] * (math.cos(2 * math.pi * k * n / N) + 1j * math.sin(2 * math.pi * k * n / N))
            temp += a
        temp /= N
        IDFTresult.append(temp)

    rounded_IDFTresult = [round(temp.real, 5) + round(temp.imag, 5) * 1j for temp in IDFTresult]

    samples = np.real(rounded_IDFTresult)
    samples = list(samples)

    indices = []
    for i in range(len(samples)):
        indices.append(i)

    if operations.get() == "DC_component_freq":
        return list(indices), list(samples)

    plotingSignal(indices, samples, 1)

def shifting_signals(isFolded):
    filepath = openFile()
    indices, samples = readfile(filepath)

    shift_amount = int(shift.get())
    indices_arr = np.array(indices)

    if(isFolded):

        inverted_indices =[]
        inverted_samples = []

        for i in range(0, len(samples)):
            inverted_indices.append(indices[len(indices) - (i+1)])
            inverted_samples.append(samples[len(samples) - (i+1)])

        indices_arr = np.array(inverted_indices)
        indices_arr *= -1
        samples = inverted_samples
        indices_arr += shift_amount

    else:
        indices_arr -= shift_amount

    indices = list(indices_arr)


    plotingSignal(indices, samples, len(samples))
    Shift_Fold_Signal(indices, samples)

def first_derivative(signal):
        y = [signal[n] - signal[n - 1] if n > 0 else signal[n] for n in range(1, len(signal))]
        return y

def second_derivative(signal):
        y = [signal[n + 1] - 2 * signal[n] + signal[n - 1] if 0 < n < len(signal) - 1 else signal[n] for n in range(1, len(signal)-1)]
        return y

def DerivativeSignal():
        InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                       18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                       34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
                       50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0,
                       82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0,
                       98.0, 99.0, 100.0]
        expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        """
        Write your Code here:
        Start
        """

        FirstDrev = first_derivative(InputSignal)
        SecondDrev = second_derivative(InputSignal)
        """
        End
        """

        """
        Testing your Code
        """

        if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
            mb.showinfo("Mismatch", "Mismatch in length.")
            return

        first, second = True, True
        for i in range(len(expectedOutput_first)):
            if abs(FirstDrev[i] - expectedOutput_first[i]) >= 0.01:
                first = False
                mb.showerror("derivative", "1st derivative wrong.")
                return

        for i in range(len(expectedOutput_second)):
            if abs(SecondDrev[i] - expectedOutput_second[i]) >= 0.01:
                second = False
                mb.showerror("derivative", "2nd derivative wrong.")
                return

        if first and second:
            mb.showinfo("Test Case Passed", "Derivative Test case passed successfully.")
        else:
            mb.showerror("Test case failed", "Derivative Test case failed")
        return

def DCT_Operation():
    filepath = openFile()
    indices, samples = readfile(filepath)
    DCTresult = []
    N = len(samples)
    for k in range(N):
        sum_val = 0
        for n in range(len(indices)):
            sum_val += samples[n] * np.cos(np.pi / (4 * N) * (2 * n - 1) * (2 * k - 1))
        DCTresult.append(round(np.sqrt(2 / N) * sum_val, 5))
        print("DCT Coefficients:")
        for k in range(len(DCTresult)):
            print(f"y({k}) = {DCTresult[k]}")

    m = simpledialog.askinteger("Input", "Enter the number of coefficients to save:")
    if m > len(DCTresult):
        mb.showerror("Invalid Input", "Please enter a valid number of coefficients.")
        return

    saveFilepath = asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if saveFilepath:
        with open(saveFilepath, "w") as f:
            for i in range(m):
                f.write(f"{DCTresult[i]}\n")
        mb.showinfo("Success", f"First {m} DCT coefficients saved to {saveFilepath}")

    fig1, ax1 = plt.subplots()
    ax1.stem(indices, DCTresult)
    ax1.set_xlabel("Coefficient Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("DCT Coefficients")

    fig1, ax1 = plt.subplots()
    ax1.plot(indices, DCTresult)
    ax1.set_xlabel("Coefficient Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("DCT Coefficients")
    plt.show()
    SignalAreEqual(DCTresult)

def DC_component_time_domain():
    filepath = openFile()
    indices, samples = readfile(filepath)
    mean_value = np.mean(samples)
    dc_component = (samples - mean_value)
    SignalSamplesAreEqual(indices, dc_component)


def DC_component_freq_domain():
    index_dft, sampledft = DFT_Operation()

    counter = 0
    for i in index_dft:
        if i == 0:
            break
        else:
            counter += 1

    sampledft[counter] = 0

    angle_list = []
    amplitude_list = []

    for i in range(len(sampledft)):
        angle = math.atan2(np.imag(sampledft[i]),  np.real(sampledft[i]))
        amplitude = math.sqrt(math.pow(np.real(sampledft[i]), 2) + math.pow(np.imag(sampledft[i]), 2))
        amplitude_list.append(amplitude)
        angle_list.append(angle)

    index, sample = IDFT_Operation(amplitude_list, angle_list)
    SignalSamplesAreEqual(index, sample)


def moving_average():
    filepath = openFile()
    indices, samples = readfile(filepath)

    window_size_str = simpledialog.askstring("Input", "Enter the number of points for averaging (window size):")

    try:
        window_size = int(window_size_str)

        if window_size <= 0:
            print("Window size must be a positive integer.")
            return
        elif window_size > len(samples):
            print("Window size must be less than or equal to the number of samples.")
            return

    except ValueError:
        print("Invalid input. Please enter a valid integer.")
        return

    output_length = len(samples) - window_size + 1
    y = []

    for n in range(output_length):
        avg_value = np.mean(samples[n:n + window_size])
        y.append(avg_value)

    output_indices = indices[:output_length]

    fig, ax = plt.subplots()
    ax.plot(indices, samples, label="Original Signal", color='blue')
    ax.plot(output_indices, y, label="Smoothed Signal (Moving Average)", color='red', linestyle='--')
    ax.set_xlabel('Index')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal Smoothing: Moving Average')
    ax.legend()
    plt.show()
    SignalSamplesAreEqual(indices, y)


def Crosscorrelation():
    filepath1 = openFile()
    signal1_indices, signal1_samples = readfile(filepath1)
    filepath2 = openFile()
    signal2_indices, signal2_samples = readfile(filepath2)
    N = len(signal1_samples)
    p12 = []

    for j in range(N):
        rotated_signal2 = signal2_samples[j:] + signal2_samples[:j]
        sum_val = sum(signal1_samples[n] * rotated_signal2[n] for n in range(N))
        signal1 = sum(signal1_samples[n] ** 2 for n in range(N))
        signal2 = sum(rotated_signal2[n] ** 2 for n in range(N))
        denominator = np.sqrt(signal1 * signal2)
        p12.append(round(sum_val / denominator, 8))
    Compare_Signals(signal1_indices, p12)

def Convolution(indices1=[], h=[]):

    if operations.get() == "FIR":
        filepath = openFile()
        indices_signal1, samples_signal1 = indices1, h
        indices_signal2, samples_signal2 = readfile(filepath)

    else:
        filepath = openFile()
        indices_signal1, samples_signal1 = readfile(filepath)
        filepath = openFile()
        indices_signal2, samples_signal2 = readfile(filepath)

    start = indices_signal1[0] + indices_signal2[0]
    end = indices_signal1[-1] + indices_signal2[-1]

    con_result_indices = []
    con_result_samples = []

    for i in range(start, end + 1):
        conv_sum = 0
        for j in range(len(samples_signal1)):
            for k in range(len(samples_signal2)):
                if indices_signal1[j] + indices_signal2[k] == i:
                    conv_sum += samples_signal1[j] * samples_signal2[k]

        con_result_indices.append(i)
        con_result_samples.append(conv_sum)

    if operations.get() == "FIR":
        return list(con_result_indices), list(con_result_samples)

    fig1, ax1 = plt.subplots()
    ax1.stem(con_result_indices, con_result_samples)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Digital Signal")

    fig2, ax2 = plt.subplots()
    ax2.plot(con_result_indices, con_result_samples)
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Analog Signal")
    ax2.axhline(0, color='black', linewidth=1)
    plt.show()

    ConvTest(con_result_indices, con_result_samples)


def ConvTest(Your_indices, Your_samples):
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]

    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """

    expected_indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        mb.showerror("Conv Test case failed", "your signal have different length from the expected one.")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            mb.showerror("Conv Test case failed", "your signal have different indicies from the expected one.")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            mb.showerror("Conv Test case failed", "your signal have different values from the expected one.")
            return
    mb.showinfo("Test Case Passed", "Conv Test case passed successfully")


def FIR_filter():

    fs = float(simpledialog.askstring("Input", "Enter the sampling frequency (Hz):"))
    filter_type = simpledialog.askstring("Input", "Enter filter type (low, high, bandpass, bandstop):").lower()
    stop_attenuation = float(simpledialog.askstring("Input", "Enter the stop band attenuation (δs):"))
    transition_band = float(simpledialog.askstring("Input", "Enter the transition band (Hz):"))

    if filter_type in ["low", "high"]:
        cutoff_freq = float(simpledialog.askstring("Input", "Enter the cutoff frequency (Hz):"))
    elif filter_type in ["bandpass", "bandstop"]:
        f1 = float(simpledialog.askstring("Input", "Enter the lower cutoff frequency (Hz):"))
        f2 = float(simpledialog.askstring("Input", "Enter the upper cutoff frequency (Hz):"))
    else:
        mb.showerror("Error", "Invalid filter type!")
        return

    delta_f_normalized = transition_band / fs
    if filter_type in ["low"]:
        cutoff = (cutoff_freq + (transition_band / 2)) / fs
    elif filter_type in ["high"]:
        cutoff = (cutoff_freq - (transition_band / 2)) / fs
    elif filter_type in ["bandpass"]:
        cutoff_1 = (f1 - (transition_band / 2)) / fs
        cutoff_2 = (f2 + (transition_band / 2)) / fs
    else:
        cutoff_1 = (f1 + (transition_band / 2)) / fs
        cutoff_2 = (f2 - (transition_band / 2)) / fs

    if stop_attenuation <= 21:
        constant = 0.9
        window_name = "Rectangular"
    elif stop_attenuation <= 44:
        constant = 3.1
        window_name = "Hanning"
    elif stop_attenuation <= 53:
        constant = 3.3
        window_name = "Hamming"
    else:
        constant = 5.5
        window_name = "Blackman"

    N = int(np.ceil(constant / delta_f_normalized))
    if N % 2 == 0:
        N += 1
    middle = N // 2

    h_d = np.zeros(N)
    if filter_type == "low":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 2 * cutoff
            else:
                h_d[n + middle] = np.sin(2 * np.pi * cutoff * n) / (np.pi * n)
    elif filter_type == "high":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 1 - (2 * cutoff)
            else:
                h_d[n + middle] = -(np.sin(2 * np.pi * cutoff * n) / (np.pi * n))
    elif filter_type == "bandpass":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 2 * (cutoff_2 - cutoff_1)
            else:
                h_d[n + middle] = (np.sin(2 * np.pi * cutoff_2 * n) - np.sin(2 * np.pi * cutoff_1 * n)) / (np.pi * n)
    elif filter_type == "bandstop":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 1 - 2 * (cutoff_2 - cutoff_1)
            else:
                h_d[n + middle] = (np.sin(2 * np.pi * cutoff_1 * n) - np.sin(2 * np.pi * cutoff_2 * n)) / (np.pi * n)

    n = np.arange(-middle, middle + 1)
    if window_name == "Rectangular":
        window = 1
    elif window_name == "Hanning":
        window = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
    elif window_name == "Hamming":
        window = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
    elif window_name == "Blackman":
        window = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

    h = h_d * window

    indices = np.arange(-middle, middle + 1)
    coefficients_with_indices = np.column_stack((indices, h))
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if save_path:
        np.savetxt(save_path, coefficients_with_indices, fmt=["%d", "%.10f"])
        mb.showinfo("Success", f"Filter coefficients saved to {save_path}")

    input_file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not input_file:
        mb.showerror("Error", "Input signal file not provided!")
        return

    fig, ax = plt.subplots()
    ax.plot(indices, h)
    ax.stem(indices, h)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original vs. Filtered Signal")
    ax.legend()
    plt.show()

    Compare_Signals(indices, h)

def FIR():

    fs = float(simpledialog.askstring("Input", "Enter the sampling frequency (Hz):"))
    filter_type = simpledialog.askstring("Input", "Enter filter type (low, high, bandpass, bandstop):").lower()
    stop_attenuation = float(simpledialog.askstring("Input", "Enter the stop band attenuation (δs):"))
    transition_band = float(simpledialog.askstring("Input", "Enter the transition band (Hz):"))

    if filter_type in ["low", "high"]:
        cutoff_freq = float(simpledialog.askstring("Input", "Enter the cutoff frequency (Hz):"))
    elif filter_type in ["bandpass", "bandstop"]:
        f1 = float(simpledialog.askstring("Input", "Enter the lower cutoff frequency (Hz):"))
        f2 = float(simpledialog.askstring("Input", "Enter the upper cutoff frequency (Hz):"))
    else:
        mb.showerror("Error", "Invalid filter type!")
        return

    delta_f_normalized = transition_band / fs
    if filter_type in ["low"]:
        cutoff = (cutoff_freq + (transition_band / 2)) / fs
    elif filter_type in ["high"]:
        cutoff = (cutoff_freq - (transition_band / 2)) / fs
    elif filter_type in ["bandpass"]:
        cutoff_1 = (f1 - (transition_band / 2)) / fs
        cutoff_2 = (f2 + (transition_band / 2)) / fs
    else:
        cutoff_1 = (f1 + (transition_band / 2)) / fs
        cutoff_2 = (f2 - (transition_band / 2)) / fs

    if stop_attenuation <= 21:
        constant = 0.9
        window_name = "Rectangular"
    elif stop_attenuation <= 44:
        constant = 3.1
        window_name = "Hanning"
    elif stop_attenuation <= 53:
        constant = 3.3
        window_name = "Hamming"
    else:
        constant = 5.5
        window_name = "Blackman"

    N = int(np.ceil(constant / delta_f_normalized))
    if N % 2 == 0:
        N += 1
    middle = N // 2

    h_d = np.zeros(N)
    if filter_type == "low":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 2 * cutoff
            else:
                h_d[n + middle] = np.sin(2 * np.pi * cutoff * n) / (np.pi * n)
    elif filter_type == "high":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 1 - (2 * cutoff)
            else:
                h_d[n + middle] = -(np.sin(2 * np.pi * cutoff * n) / (np.pi * n))
    elif filter_type == "bandpass":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 2 * (cutoff_2 - cutoff_1)
            else:
                h_d[n + middle] = (np.sin(2 * np.pi * cutoff_2 * n) - np.sin(2 * np.pi * cutoff_1 * n)) / (np.pi * n)
    elif filter_type == "bandstop":
        for n in range(-middle, middle + 1):
            if n == 0:
                h_d[n + middle] = 1 - 2 * (cutoff_2 - cutoff_1)
            else:
                h_d[n + middle] = (np.sin(2 * np.pi * cutoff_1 * n) - np.sin(2 * np.pi * cutoff_2 * n)) / (np.pi * n)

    n = np.arange(-middle, middle + 1)
    if window_name == "Rectangular":
        window = 1
    elif window_name == "Hanning":
        window = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
    elif window_name == "Hamming":
        window = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
    elif window_name == "Blackman":
        window = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

    h = h_d * window

    indices1 = np.arange(-middle, middle + 1)

    index, sample = Convolution(indices1, h)

    coefficients_with_indices = np.column_stack((index, sample))
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if save_path:
        np.savetxt(save_path, coefficients_with_indices, fmt=["%d", "%.10f"])
        mb.showinfo("Success", f"Filter coefficients saved to {save_path}")

    input_file = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not input_file:
        mb.showerror("Error", "Input signal file not provided!")
        return

    fig, ax = plt.subplots()
    ax.plot(index, sample)
    # ax.stem(index, sample)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Original vs. Filtered Signal")
    ax.legend()
    plt.show()

    Compare_Signals(index, sample)

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

    elif operations.get() == "DFT":
        DFT_Operation()
    elif operations.get() == "IDFT":
        IDFT_Operation()
    elif operations.get() == "Fold" or operations.get() == "Shift":
        shifting_signals(True if operations.get() == "Fold" else False)

    elif operations.get() == "Sharpening":
        DerivativeSignal()
    elif operations.get() == "DCT":
        DCT_Operation()
    elif operations.get() == "correlation":
        Crosscorrelation()
    elif operations.get() == "DC_component_time":
        DC_component_time_domain()
    elif operations.get() == "DC_component_freq":
        DC_component_freq_domain()
    elif operations.get() == "moving_average":
        moving_average()
    elif operations.get() == "Convolution":
        Convolution()
    elif operations.get() == "FIR_filters":
        FIR_filter()
    elif operations.get() == "FIR":
        FIR()

def plotingSignal(indices, samples, samplingFrequency):
    indicesArr = np.array(indices)
    samplesArr = np.array(samples)

    fig1, ax1 = plt.subplots()
    ax1.stem(indicesArr, samplesArr)
    ax1.set_xlim(indicesArr[0], indicesArr[0] + samplingFrequency * 0.1)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Digital Signal")

    timeArr = indicesArr / samplingFrequency
    fig2, ax2 = plt.subplots()
    ax2.plot(timeArr, samplesArr)
    ax2.set_xlim(timeArr[0], timeArr[0] + 0.1)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Analog Signal")
    ax2.axhline(0, color='black', linewidth=1)
    plt.show()

    if(operations.get() != "Quantize" and operations.get() != "Fold" and operations.get() != "Shift" ):
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

operations = ttk.Combobox(myframe, values=["None", "Add", "Subtract", "Multiply", "Square",
                                           "Normalize", "Accumulate", "Quantize", "DFT", "IDFT",
                                           "Fold", "Shift", "Sharpening", "DCT", "correlation",
                                           "Convolution", "DC_component_time", "DC_component_freq",
                                           "moving_average", "FIR_filters", "FIR"], width=47)
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

mylab = ttk.Label(myframe, text="Enter The Shifting Amount")
mylab.place(relx=0.5, rely=0.5, x=-450, y=195, anchor="center")

shift = ttk.Entry(myframe, width=50)
shift.insert(END, '0')
shift.place(relx=0.5, rely=0.5, x=-450, y=220, anchor="center")

mybottom =ttk.Button(myframe,text="Display Result", width=50, command=generateSignal)
mybottom.place(relx=0.5, rely=0.5, y=280,anchor="center")

mybottom2 = ttk.Button(myframe, text="Read File", width=50, command=readFileDirct)
mybottom2.place(relx=0.5, rely=0.5, x=400, y=280, anchor="center")

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