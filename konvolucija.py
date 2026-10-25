import numpy as np
from PIL import Image

def paddaj(A,padding):
    for i in range(len(A)):
        for j in range(padding):
            A[i].insert(0,0)
            A[i].append(0)
    nul_redak = [0 for _ in range(len(A)+2*padding)]

    for i in range(padding):
        A.insert(0,nul_redak)
        A.append(nul_redak)

def konvolucija_matrica(A,B,padding,stride):
    if(padding>0):
        paddaj(A,padding)
    else:
        print("Ne treba paddat")
        
    dimenzija_a=len(A[0])
    dimenzija_b=len(B[0])

    dimenzija_c=(dimenzija_a-dimenzija_b)//stride+1

    C=[[0 for i in range(dimenzija_c)] for j in range(dimenzija_c)]

    offset=dimenzija_b//2
    try:
        for i in range(offset,dimenzija_a,stride):
            try:
                for j in range(offset,dimenzija_a,stride):
                    suma=0
                    for k in range(dimenzija_b):
                        for l in range(dimenzija_b):
                            suma+=A[i+k-offset][j+l-offset]*B[k][l]
                    C[(i-offset)//stride][(j-offset)//stride]=suma
                    
            except IndexError as e:
                print(f"Caught an out-of-bounds error: {e} = {j}")
    except IndexError as e:
        print(f"Caught an out-of-bounds error: {e} {i}")
    return C
            
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0]
], dtype=np.int32)

img = Image.fromarray((A*255).astype(np.uint8), mode='L')
img_scaled = img.resize((256, 256), Image.NEAREST)
img_scaled.show()

A =(np.array(img_scaled, dtype=np.int32)//255)

max_abs = max(abs(A.min()), abs(A.max()))
if max_abs == 0:
    max_abs = 1
normalized = (A + max_abs) / (2 * max_abs)
img_array = (normalized * 255).astype(np.uint8)
imageA = Image.fromarray(img_array, mode='L')
#imageA.show()

B_1 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
],dtype=np.int32)

B_2 = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
],dtype=np.int32)

C1=konvolucija_matrica(A,B_1,0,1)
C1_arr = np.array(C1, dtype=np.float32)

C2=konvolucija_matrica(A,B_2,0,1)
C2_arr = np.array(C2, dtype=np.float32)

max_abs = max(abs(C1_arr.min()), abs(C1_arr.max()))
if max_abs == 0:
    max_abs = 1
normalized = (C1_arr + max_abs) / (2 * max_abs)
img_array = (normalized * 255).astype(np.uint8)

image1 = Image.fromarray(img_array, mode='L')

max_abs = max(abs(C2_arr.min()), abs(C2_arr.max()))
if max_abs == 0:
    max_abs = 1
normalized = (C2_arr + max_abs) / (2 * max_abs)
img_array = (normalized * 255).astype(np.uint8)

image2 = Image.fromarray(img_array, mode='L') 

image1.show()
image2.show()
