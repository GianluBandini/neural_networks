import re
#text to train on
with open('Transcript.txt', 'r',encoding = 'utf-8') as f:
    text = f.read()

#unique characters in the text and the number of them
chars = sorted(list(set(text)))
print(chars)
print('------------------')
chars = chars[:19]+ chars[46:48] + chars[74:]
print('------------------')
print(chars[46:48])
print('------------------')
print(chars[74:])
chars += sorted(set(re.findall(r'\S+', text)))

