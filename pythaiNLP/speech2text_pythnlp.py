# -*- coding: utf-8 -*-
"""speech2text_pyTHNLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cwuMG19qsmPF0Q0XeHXHvOswi_S67j1F
"""

!git clone https://github.com/wannaphong/NeMo.git

cd NeMo

!pip install -q -e .[tts]

!pip install -q pythaitts

from pythaitts import TTS

tts = TTS(pretrained="lunarlist")

import IPython

file = tts.tts("ภาษาไทย ง่าย มาก มาก",filename="cat.wav")

IPython.display.Audio(file)

wave = tts.tts("ภาษาไทย ง่าย มาก มาก",return_type="waveform")
IPython.display.Audio(wave, rate=22010) # load a NumPy array

file = tts.tts("ภาษาไทยตัดคำได้นานแล้ว")

IPython.display.Audio(file)

file = tts.tts("ภาษา ไทย ตัด คำ ได้ นาน แล้ว")

IPython.display.Audio(file)

file = tts.tts("ปพ")

IPython.display.Audio(file)

num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_read =["หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]

# Define a dictionary to map digits to their Thai word equivalents
digit_to_thai_word = {
    '0': 'ศูนย์',
    '1': 'หนึ่ง',
    '2': 'สอง',
    '3': 'สาม',
    '4': 'สี่',
    '5': 'ห้า',
    '6': 'หก',
    '7': 'เจ็ด',
    '8': 'แปด',
    '9': 'เก้า'
}

# Input string
input = "ปพ1234 กรุงเทพมหานคร"
input_string,pv = input.split(" ")

# Initialize an empty result string
result_string = ""
o = "อ"
# Iterate through the input string and map digits to Thai words
for char in input_string:
    if char.isdigit():
        result_string += digit_to_thai_word[char] + ' '
    else:
        result_string += char
        result_string += o
        result_string += ' '

# Remove trailing space
result_string = result_string.strip()
result_string += ' '
result_string += pv

# Output the result
print(result_string)