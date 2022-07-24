#imports
from re import MULTILINE
from typing_extensions import Self
from matplotlib.image import PIL
from PIL import Image
import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment
import numpy as np
import scipy
import matplotlib.pyplot as plt
import array
from collections import Counter
import numpy as np
import scipy
from pydub.utils import get_array_type
from Levenshtein import distance
import array
from pydub import AudioSegment
from pydub.utils import get_array_type
from kivy.app import App
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
import soundfile as sf
import threading
import soundfile as sf
import sounddevice as sd
import queue
import soundfile as sf
import threading

#code

NOTES = {
    "C0": 16.35,
    "C#0": 17.32,
    "D0": 18.35,
    "D#0": 19.45,
    "E0": 20.60,
    "F0": 21.83,
    "F#0": 23.12,
    "G0": 24.50,
    "G#0": 25.96,
    "A0": 27.5,
    "A#0": 29.14,
    "B0": 30.87, 
    "C1": 32.70,
    "C#1": 34.65,
    "D1": 36.71,
    "D#1": 38.89,
    "E1": 41.20,
    "F1": 43.65,
    "F#1": 46.25,
    "G1": 49.00,
    "G#1": 51.91,
    "A1": 55.00,
    "A#1": 58.27,
    "B1": 61.74, 
    "C2": 65.41,
    "C#2": 69.30,
    "D2": 73.42,
    "D#2": 77.78,
    """  
    --0--
    -----
    -----
    -----
    -----
    -----""": 82.41,#E2
    """ 
    --1--
    -----
    -----
    -----
    -----
    ----""": 87.31,#F2
    """ 
    --2--
    -----
    -----
    -----
    -----
    -----""": 92.50,#F#2
    """
    --3--
    -----
    -----
    -----
    -----
    -----""": 98.00,#G2 
    """
    --4--
    -----
    -----
    -----
    -----
    -----""": 103.83,#G#2 
    """
    -----
    --0--
    -----
    -----
    -----
    -----""": 110.00,#A2 
    """
    -----
    --1--
    -----
    -----
    -----
    -----""": 116.54,#A#2 
    """ 
    -----
    --2--
    -----
    -----
    -----
    -----""": 123.47,#B2
    """
    -----
    --3--
    -----
    -----
    -----
    -----""": 130.81,#C3 
    """ 
    -----
    --4--
    -----
    -----
    -----
    -----""": 138.59,#C#3
    """ 
    -----
    -----
    --0--
    -----
    -----
    -----""": 146.83,#D3
    """
    -----
    -----
    --1--
    -----
    -----
    -----""": 155.56,#D#3 
    """ 
    -----
    -----
    --2--
    -----
    -----
    -----
    """: 164.81,#E3
    """ 
    -----
    -----
    --3--
    -----
    -----
    -----""": 174.61,#F3
    """
    -----
    -----
    --4--
    -----
    -----
    -----""": 185.00,#F#3 
    """ 
    -----
    -----
    -----
    --0--
    -----
    -----""": 196.00,#G3
    """ 
    -----
    -----
    -----
    --1--
    -----
    -----""": 207.65,#G#3
    """
    -----
    -----
    -----
    --2--
    -----
    -----
    """: 220.00,#A3 
    """
    -----
    -----
    -----
    --3--
    -----
    -----""": 233.08,#A#3
    """
    -----
    -----
    -----
    -----
    --0--
    -----""": 246.94,#B3 
    """
    -----
    -----
    -----
    -----
    --1--
    -----""": 261.63,#C4 
    """ 
    -----
    -----
    -----
    -----
    --2--
    -----""": 277.18,#C#4
    """
    -----
    -----
    -----
    -----
    --3--
    -----""": 293.66,#D4 
    """
    -----
    -----
    -----
    -----
    -----
    --0--""": 311.13,#D#4 
    """
    -----
    -----
    -----
    -----
    -----
    --1--""": 329.63,#E4 
    """ 
    -----
    -----
    -----
    -----
    -----
    --2--""": 349.23,#F4
    """ 
    -----
    -----
    -----
    -----
    -----
    --3--""": 369.99,#F#4
    """ 
    -----
    -----
    -----
    -----
    -----
    --4--""": 392.00,#G4
    """ 
    -----
    -----
    -----
    -----
    -----
    --5--""": 415.30,#G#4
    """ 
    -----
    -----
    -----
    --15--
    -----
    -----""": 440.00,#A4
    """
    -----
    -----
    -----
    --16--
    -----
    -----""": 466.16,#A#4
    """
    -----
    -----
    -----
    --17--
    -----
    -----""": 493.88,#B4
    """ 
    -----
    -----
    -----
    --18--
    -----
    -----""": 523.25,#C5
    """ 
    -----
    -----
    -----
    -----
    --15--
    -----""": 554.37,#C#5
    """ 
    -----
    -----
    -----
    -----
    --16--
    -----""": 587.33,#D5
    """
    -----
    -----
    -----
    -----
    --17--
    -----""": 622.25,#D#5
    """ 
    -----
    -----
    -----
    -----
    --18--
    -----""": 659.25,#E5
    """ 
    -----
    -----
    -----
    -----
    -----
    --14--""": 698.46,#F5
    """ 
    -----
    -----
    -----
    -----
    -----
    --15--""": 739.99,#F#5
    """
    -----
    -----
    -----
    -----
    -----
    --16--""": 783.99,#G5
    """ 
    -----
    -----
    -----
    -----
    -----
    --17--""": 830.61,#G#5
    """ 
    -----
    -----
    -----
    -----
    -----
    --18--""": 880.00,#A5
    """ 
    -----
    -----
    -----
    -----
    -----
    --19--""": 932.33,#A#5
    """ 
    -----
    -----
    -----
    -----
    -----
    --20--""": 987.77,#B5
    """ 
    -----
    -----
    -----
    -----
    -----
    --21--""": 1046.50,#C6
    """ 
    -----
    -----
    -----
    -----
    -----
    --22--""": 1108.73,#C#6
    """ 
    -----
    -----
    -----
    -----
    -----
    --23--""": 1174.66,#D6
    """ 
    -----
    -----
    -----
    -----
    -----
    --24--""": 1244.51,#D#6
    "E6": 1318.51,
    "F6": 1396.91,
    "F#6": 1479.98,
    "G6": 1567.98,
    "G#6": 1661.22,
    "A6": 1760.00,
    "A#6": 1864.66,
    "B6": 1975.53,
    "C7": 2093.00,
    "C#7": 2217.46,
    "D7": 2349.32,
    "D#7": 2489.02,
    "E7": 2637.02,
    "F7": 2793.83,
    "F#7": 2959.96,
    "G7": 3135.96,
    "G#7": 3322.44,
    "A7": 3520.00,
    "A#7": 3729.31,
    "B7": 3951.07,
    "C8": 4186.01,
    "C#8": 4434.92,
    "D8": 4698.63,
    "D#8": 4978.03,
    "E8": 5274.04,
    "F8": 5587.65,
    "F#8": 5919.91,
    "G8": 6271.93,
    "G#8": 6644.88,
    "A8": 7040.00,
    "A#8": 7458.62,
    "B8": 7902.13,
}

def main(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
    
  
    record_file()
    
    # If a note file and/or actual start times are supplied read them in
    actual_starts = []
    if note_starts_file:
        with open(note_starts_file) as f:
            for line in f:
                actual_starts.append(float(line.strip()))

    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())

    song = AudioSegment.from_file(file)
    song = song.high_pass_filter(80, order=4)

    starts = predict_note_starts(song, plot_starts, actual_starts)

    predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)

    
    print("Predicted Notes")
    print(*predicted_notes, sep='\n')

    

    #kivy stuff, work in progress

    #tabs_string =''.join([str(item) for item in predicted_notes])

    #class MyGrid(GridLayout):
        #def __init__(self, **kwargs):
            #super(MyGrid, self).__init__(**kwargs)
            #self.cols =1
            #self.add_widget(Label(text=tabs_string))
            #self.name = TextInput(multiline=False)
             

    #class MyApp(App):
        #def build(self):
            #return MyGrid()

    #if __name__ == "__main__":
        #MyApp().run()

    #/////

    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))

def record_file():
      #python-sounddevice - recording audio
    fs = 48000  # Sample rate
    seconds = 5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Recording for " + str(seconds) + " seconds!")
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 

def play_recording():
    playsound('output.wav')

def predict_note_starts(song, plot, actual_starts):
    # Size of segments to break song into for volume calculations
    SEGMENT_MS = 50
    # Minimum volume necessary to be considered a note
    VOLUME_THRESHOLD = -35
    # The increase from one sample to the next required to be considered a note
    EDGE_THRESHOLD = 5
    # Throw out any additional notes found in this window
    MIN_MS_BETWEEN = 100

    # Filter out lower frequencies to reduce noise
    song = song.high_pass_filter(80, order=4)
    # dBFS is decibels relative to the maximum possible loudness
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD:
            ms = i * SEGMENT_MS
            # Ignore any too close together
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    # If actual note start times are provided print a comparison
    #if len(actual_starts) > 0:
        #print("Approximate actual note start times ({})".format(len(actual_starts)))
        #print(" ".join(["{:5.2f}".format(s) for s in actual_starts]))
        #print("Predicted note start times ({})".format(len(predicted_starts)))
        #print(" ".join(["{:5.2f}".format(ms // 1000) for ms in predicted_starts]))

    # Plot the volume over time (sec)
    if plot:
        x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
        plt.plot(x_axis, volume)

        # Add vertical lines for predicted note starts and actual note starts
        for s in actual_starts:
            plt.axvline(x=s, color="r", linewidth=0.5, linestyle="-")
        for ms in predicted_starts:
            plt.axvline(x=(ms // 1000), color="g", linewidth=0.5, linestyle=":")

        plt.show()

    return predicted_starts

def predict_notes(song, starts, actual_notes, plot_fft_indices):
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_spectrum(segment)

        predicted = classify_note_attempt_2(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U")

        # Print general info
        #print("")
        #print("Note: {}".format(i))
        #if i < len(actual_notes):
            #print("Predicted: {} Actual: {}".format(predicted, actual_notes[i]))
        #else:
            #print("Predicted: {}".format(predicted))
        #print("Predicted start: {}".format(start))
        #length = sample_to - sample_from
        #print("Sampled from {} to {} ({} ms)".format(sample_from, sample_to, length))
        #print("Frequency sample period: {}hz".format(freqs[1]))

        # Print peak info
        peak_indicies, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        #print("Peaks of more than 1.5 percent of total frequency contribution:")
        for j, peak in enumerate(peak_indicies):
            freq = freqs[peak]
            magnitude = props["peak_heights"][j]
            #print("{:.1f}hz with magnitude {:.3f}".format(freq, magnitude))

        if i in plot_fft_indices:
            plt.plot(freqs, freq_magnitudes, "b")
            plt.xlabel("Freq (Hz)")
            plt.ylabel("|X(freq)|")
            plt.show()
    return predicted_notes

def frequency_spectrum(song, max_frequency=8000):
    """
    Derive frequency spectrum of a signal pydub.AudioSample
    Returns an array of frequencies and an array of how prevelant that frequency is in the sample
    """
    # Convert pydub.AudioSample to raw audio data
    # Copied from Jiaaro's answer on https://stackoverflow.com/questions/32373996/pydub-raw-audio-data

    bit_depth = song.sample_width * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, song._data)
    n = len(raw_audio_data)
    if array_type == 0:
        raise Exception("array_type length is 0, value should not be zero")

    # Compute FFT and frequency value for each index in FFT array
    # Inspired by Reveille's answer on https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
    freq_array = np.arange(n) * (float(song.frame_rate) / n)  # two sides frequency range
    freq_array = freq_array[: (n // 2)]  # one side frequency range

    raw_audio_data = raw_audio_data - np.average(raw_audio_data)  # zero-centering
    freq_magnitude = scipy.fft.fft(raw_audio_data)  # fft computing and normalization
    freq_magnitude = freq_magnitude[: (n // 2)]  # one side

    if max_frequency:
        max_index = int(max_frequency * n / song.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    freq_magnitude = abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    return freq_array, freq_magnitude

def classify_note_attempt_1(freq_array, freq_magnitude):
    i = np.argmax(freq_magnitude)
    f = freq_array[i]
    #print("frequency {}".format(f))
    #print("magnitude {}".format(freq_magnitude[i]))
    return get_note_for_freq(f)

def classify_note_attempt_2(freq_array, freq_magnitude):
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue
        note = get_note_for_freq(freq_array[i])
        if note:
            note_counter[note] += freq_magnitude[i]
    return note_counter.most_common(1)[0][0]

def classify_note_attempt_3(freq_array, freq_magnitude):
    min_freq = 0
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue

        for freq_multiplier, credit_multiplier in [
            (1, 1),
            (1 / 3, 3 / 4),
            (1 / 5, 1 / 2),
            (1 / 6, 1 / 2),
            (1 / 7, 1 / 2),
        ]:
            freq = freq_array[i] * freq_multiplier
            if freq < min_freq:
                continue
            note = get_note_for_freq(freq)
            if note:
                note_counter[note] += freq_magnitude[i] * credit_multiplier

    return note_counter.most_common(1)[0][0]

def get_note_for_freq(f, tolerance=33):
    # Calculate the range for each note
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {
        k: (v / tolerance_multiplier, v * tolerance_multiplier) for (k, v) in NOTES.items()
    }
    

    # Check if any notes match
    for (note, note_range) in note_ranges.items():
        if f > note_range[0] and f < note_range[1]:
            return note
    return None

def calculate_distance(predicted, actual):
    # To make a simple string for distance calculations we make natural notes lower case
    # and sharp notes cap
    def transform(note):
        if "#" in note:
            return note[0].upper()
        return note.lower()

    return distance(
        "".join([transform(n) for n in predicted]), "".join([transform(n) for n in actual]),
    )
      
class MainWindow(Screen):
    Window.clearcolor = (100,100,100)
q = queue.Queue()
recording = False
file_exists = False    

def callback(indata, frames, time, status):
    q.put(indata.copy())

def record_audio():
          #Declare global variables    
        global recording 
        #Set to True to record
        recording= True   
        global file_exists 
        #Create a file to save the audio
        print('recording audio')
        with sf.SoundFile("output.wav", mode='w', samplerate=44100,
                            channels=1) as file:
        #Create an input stream to record audio without a preset time
                with sd.InputStream(samplerate=44100, channels=1, callback=callback):
                    while recording == True:
                        #Set the variable to True to allow playing the audio later
                        file_exists =True
                        #write into file
                        file.write(q.get())

class SecondWindow(Screen):
    import sounddevice as sd
    import queue
    import soundfile as sf
    

    file = 'output.wav'

    def start_recording(self):
        t1=threading.Thread(target= record_audio)
        t1.start()

    def stop_recording(self):
        #To stop, set the flag to false
        global recording
        recording = False
        print('recording finished')

    def replay_recording(self):
            #To play a recording, it must exist.
        if file_exists:
        #Read the recording if it exists and play it
            data, fs = sf.read("output.wav", dtype='float32') 
            sd.play(data,fs)
        else:
            #Display and error if none is found
            print('You need to record something silly')

class ThirdWindow(Screen):
    pass

class FourthWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")
       
class MyMainApp(App):
    def build(self):
        return kv
    
    def mainyes(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
        from pydub import AudioSegment
        file = 'output.wav'
        
    
    
        actual_starts = []
        if note_starts_file:
            with open(note_starts_file) as f:
                for line in f:
                    actual_starts.append(float(line.strip()))

        actual_notes = []
        if note_file:
            with open(note_file) as f:
                for line in f:
                    actual_notes.append(line.strip())

        song = AudioSegment.from_file(file)
        song = song.high_pass_filter(20, order=4)

        starts = predict_note_starts(song, plot_starts, actual_starts)

        predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)


        new_items = [list(filter(lambda x: x, i.splitlines())) for i in predicted_notes] # splitting the lines
        print('\n'.join(''.join(a) for a in zip(*new_items)))
        new_tabs = ('\n'.join(''.join(a) for a in zip(*new_items)))
        
        predictednotes = str(predicted_notes)

        print("Predicted Notes")
    
        print(new_tabs)
    

        #popup = Popup(title='Tabs',content=Label(text=tabs_string),size_hint=(None, None), size=(400, 400), )
            

        #popup.open()

        #def popup(instance):
            #layout_popup = GridLayout(cols=1, spacing=10, size_hint_y=None, text = tabs_string)
            #layout_popup.bind(minimum_height=layout_popup.setter('height'))

           # for i in range(0, 15):
               # btn1 = Button(text=str(i), id=str(i))
               # layout_popup.add_widget(btn1)

          #  root = ScrollView(size_hint=(1, None), size=(Window.width, Window.height))
           # root.add_widget(layout_popup)
          #  popup = Popup(title='Numbers', content=root, size_hint=(1, 1))
           # popup.open()

        def popup_display(self, title, message, widget):
            l = Label(text=message)
            l.bind(size=lambda s, w: s.setter('text_size')(s, w))

            popup = Popup(content=l, title=title, size_hint=(None, None), size=(300, 200), auto_dismiss=True)

            root = ScrollView(size_hint=(1, None), size=(Window.width, Window.height))
            root.add_widget(Label(text = message))
            popup = Popup(title='Tabs', content=root, size_hint=(1, 1), )
    

            popup.open()

        popup_display('x', 'Tabs!', new_tabs, 'x')

        if actual_notes:
            lev_distance = calculate_distance(predicted_notes, actual_notes)
            print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes))) 

if __name__ == "__main__":
    MyMainApp() .run()