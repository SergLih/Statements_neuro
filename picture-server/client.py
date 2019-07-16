import socket
import pickle
import io
import json

SIGNATURE = b'lihrazum9876543210!@#$%^&*()'
HOST = 'localhost'
PORT = 6788
ADDR = (HOST,PORT)
BUFSIZE = 4096
LS = len(SIGNATURE)
#videofile = "videos/royalty-free_footage_wien_18_640x360.mp4"

# with open("_.jpg", 'rb') as f:
# 	ba = bytearray(f.read())

g = 2
ba_info = bytearray(SIGNATURE + b'c' + g.to_bytes(4, byteorder='big')
            + b'lectures/group1/lecture_20190715_1257_lecture_g20_2017.jpg')
#ba_info = bytearray(SIGNATURE + b'u' + g.to_bytes(4, byteorder='big') + b'~/MAI/practic_2019/pictures/aud/2/2.jpg')
b = bytes(ba_info)

print(len(b))

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect(ADDR)
    client.send(b)
    print('sent')
    data = client.recv(BUFSIZE)
    d = json.loads(data[LS:].decode('ascii'))
    print('recieved: ', d)
    client.close()
except ConnectionRefusedError:
    print('[ERROR]\tServer has not responded')
