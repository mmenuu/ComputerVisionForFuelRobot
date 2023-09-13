# -*- coding: utf-8 -*-
"""phonetic_sounds.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E2eSfOJGNiu09BuPyqXC3FRNpU0EsT0H
"""

!pip install gTTS googletrans==4.0.0-rc1

from gtts import gTTS
from googletrans import Translator

# List of phonetic alphabets
phonetic_alphabets = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ']

# Initialize the Translator
translator = Translator()

# Create TTS and save audio files for each phonetic alphabet
for alphabet in phonetic_alphabets:
    translation = translator.translate(alphabet, src='en', dest='th')
    tts = gTTS(text=translation.text, lang='th')
    tts.save(f'sound_{alphabet.lower()}.mp3')
    print(f'Saved sound_{alphabet.lower()}.mp3')

from gtts import gTTS

# Define the dictionary mapping Thai characters to their phonetic transcriptions
thai_character_to_phonetic = {
    'ก': 'กอ',
    'ข': 'ขอ',
    'ฃ': 'ขอ',
    'ค': 'คอ',
    'ฅ': 'คอ',
    'ฆ': 'คอ',
    'ง': 'งอ',
    'จ': 'จอ',
    'ฉ': 'ฉอ',
    'ช': 'ชอ',
    'ซ': 'ซอ',
    'ฌ': 'ชอ',
    'ญ': 'ยอ',
    'ฎ': 'ดอ',
    'ฏ': 'ตอ',
    'ฐ': 'ถอ',
    'ฑ': 'ทอ',
    'ฒ': 'ทอ',
    'ณ': 'นอ',
    'ด': 'ดอ',
    'ต': 'ตอ',
    'ถ': 'ถอ',
    'ท': 'ทอ',
    'ธ': 'ทอ',
    'น': 'นอ',
    'บ': 'บอ',
    'ป': 'ปอ',
    'ผ': 'ผอ',
    'ฝ': 'ฝอ',
    'พ': 'พอ',
    'ฟ': 'ฟอ',
    'ภ': 'พอ',
    'ม': 'มอ',
    'ย': 'ยอ',
    'ร': 'รอ',
    'ล': 'ลอ',
    'ว': 'วอ',
    'ศ': 'สอ',
    'ษ': 'สอ',
    'ส': 'สอ',
    'ห': 'หอ',
    'ฬ': 'ลอ',
    'อ': 'ออ',
    'ฮ': 'ฮอ'
}

# Define the directory where you want to save the .mp3 files
output_directory = '/content/phonetics/'

# Iterate through the dictionary and create .mp3 files
for thai_char, phonetic_text in thai_character_to_phonetic.items():
    tts = gTTS(text=phonetic_text, lang='th')
    tts.save(f'{output_directory}{thai_char}.mp3')

print("Sound files downloaded successfully.")

!zip -r phonetics.zip /content/phonetics

from google.colab import files
i=0
for thai_char, phonetic_text in thai_character_to_phonetic.items():
    files.download(f'{output_directory}{thai_char}.mp3')
    i=i+1
print(i)