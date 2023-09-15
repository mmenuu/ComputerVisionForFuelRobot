# -*- coding: utf-8 -*-
"""manual_t2s.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13vQv5IN_UjJLx2RLJbK4pwcbVfdSB6Hv
"""

!unzip /nums.zip
!unzip /provinces.zip
!unzip /phonetics.zip

from IPython.display import Audio
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

provinces_thailand = [
    "กรุงเทพมหานคร",
    "สมุทรปราการ",
    "นนทบุรี",
    "ปทุมธานี",
    "พระนครศรีอยุธยา",
    "อ่างทอง",
    "ลพบุรี",
    "สิงห์บุรี",
    "ชัยนาท",
    "สระบุรี",
    "ชลบุรี",
    "ระยอง",
    "จันทบุรี",
    "ตราด",
    "ฉะเชิงเทรา",
    "ปราจีนบุรี",
    "นครนายก",
    "สระแก้ว",
    "นครราชสีมา",
    "บุรีรัมย์",
    "สุรินทร์",
    "ศรีสะเกษ",
    "อุบลราชธานี",
    "ยโสธร",
    "ชัยภูมิ",
    "อำนาจเจริญ",
    "บึงกาฬ",
    "หนองบัวลำภู",
    "ขอนแก่น",
    "อุดรธานี",
    "เลย",
    "หนองคาย",
    "มหาสารคาม",
    "ร้อยเอ็ด",
    "กาฬสินธุ์",
    "สกลนคร",
    "นครพนม",
    "มุกดาหาร",
    "เชียงใหม่",
    "ลำพูน",
    "ลำปาง",
    "อุตรดิตถ์",
    "แพร่",
    "น่าน",
    "พะเยา",
    "เชียงราย",
    "แม่ฮ่องสอน",
    "นครสวรรค์",
    "อุทัยธานี",
    "กำแพงเพชร",
    "ตาก",
    "สุโขทัย",
    "พิษณุโลก",
    "พิจิตร",
    "เพชรบูรณ์",
    "ราชบุรี",
    "กาญจนบุรี",
    "สุพรรณบุรี",
    "นครปฐม",
    "สมุทรสาคร",
    "สมุทรสงคราม",
    "เพชรบุรี",
    "ประจวบคีรีขันธ์",
    "นครศรีธรรมราช",
    "กระบี่",
    "พังงา",
    "ภูเก็ต",
    "สุราษฎร์ธานี",
    "ระนอง",
    "ชุมพร",
    "สงขลา",
    "สตูล",
    "ตรัง",
    "พัทลุง",
    "ปัตตานี",
    "ยะลา",
    "นราธิวาส",
]

# Input string
input = "ปพ1234 กรุงเทพมหานคร"
input_characters,pv = input.split(" ")
phonetics =[]
# for i in input_string:
#   if input_string is not None:
#     if i

for char in input_characters:
    if char in thai_character_to_phonetic:
        # If the character exists in the dictionary, add its phonetic representation to the list
        phonetics.append(char)
        audio = Audio(f'/content/content/phonetics/{char}.mp3',autoplay=True)
        audio
    elif char in digit_to_thai_word:
        # If the character is a digit, add its Thai word equivalent to the list
        phonetics.append(char)
        audio = Audio(f'/content/content/num/{char}.mp3',autoplay=True)
        audio
# Join the phonetic representations to form the final result
phonetics.append(pv)
audio = Audio(f'/content/content/provinces/{pv}.mp3',autoplay=True)
audio
result = ''.join(phonetics).join
print(phonetics)