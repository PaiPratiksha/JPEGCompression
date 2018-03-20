# import all the required libraries
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import math
import scipy.fftpack as sf


# zigzag traversal of the input array and loading it into a long row vector 

def ZigZag(Coeff):
    ZigZagMatrix = [[0,    1,  5,   6, 14,  15, 27,  28], 
                    [2,    4,  7,  13, 16,  26, 29,  42],
                    [3,    8, 12,  17, 25,  30, 41,  43],  
                    [9,   11, 18,  24, 31,  40, 44,  53],
                    [10,  19, 23,  32, 39,  45, 52,  54],  
                    [20,  22, 33,  38, 46,  51, 55,  60],  
                    [21,  34, 37,  47, 50,  56, 59,  61],  
                    [35,  36, 48,  49, 57,  58, 62,  63]]
    
    SeqBlock = np.empty(np.size(Coeff))
    BlockSize = np.shape(Coeff)
    SeqBlock = np.zeros(np.size(Coeff));
    
    for i in range(BlockSize[0]):
        for j in range(BlockSize[1]):
            SeqBlock[ZigZagMatrix[i][j]] = Coeff[i][j]
            
            
    return SeqBlock

# inverse zigzag of given sequence

def iZigZag(dSeq):
    ZigZagMatrix = [[0,    1,  5,   6, 14,  15, 27,  28], 
                [2,    4,  7,  13, 16,  26, 29,  42],
                [3,    8, 12,  17, 25,  30, 41,  43],  
                [9,   11, 18,  24, 31,  40, 44,  53],
                [10,  19, 23,  32, 39,  45, 52,  54],  
                [20,  22, 33,  38, 46,  51, 55,  60],  
                [21,  34, 37,  47, 50,  56, 59,  61],  
                [35,  36, 48,  49, 57,  58, 62,  63]]
    
    length = np.size(dSeq)
    BlockSize = (int(np.sqrt(length)),int(np.sqrt(length)))
    Coeff = np.empty((BlockSize[0], BlockSize[1]))
    for i in range(BlockSize[0]):
        for j in range(BlockSize[1]):
            Coeff[i][j] = dSeq[ZigZagMatrix[i][j]] ;
            
    return Coeff


def Reduce(SeqBlock, QNew):
    Limit = int(np.ceil(QNew * 64))
    Seq = np.empty(Limit) 
    for i in range( Limit ):
            Seq[i] = int((SeqBlock[i]));
    return Seq
    
    
    
    
    
def Expand(Seq):
    SeqBlock = np.empty(64)
    length = int(np.size(Seq))
    
    for i in range(length):
        SeqBlock[i] = int(Seq[i])
    for i in np.arange( length , 64):
        SeqBlock[i] = 0;
    return SeqBlock
    
imageInput = input('Specify the image please \n')    
ImageOriginal = cv2.imread(imageInput)
Image = cv2.cvtColor(ImageOriginal, cv2.COLOR_BGR2GRAY)
Image = np.asarray(Image, dtype = np.float64)
print("Image")
print(Image)
Image = Image - 127
m, n = np.shape(Image)
mNew, nNew = int(m/8), int(n/8)
iImage = np.zeros(np.shape(Image))
print("iImage")
print(iImage)

Q = int(float(input("Please input the quality factor\n")))

QNew = Q/100

for i in range(int(m/8)):
    
    for j in range(int(n/8)):
        
        x = Image[ (i*8) : ((i*8)+8)  , (j*8) : ((j*8)+8)]
        Dct1 = sf.dct(x, axis  = 0, norm = 'ortho')
        Dct2 = sf.dct(Dct1, axis  = 1, norm = 'ortho')      
        Coeff = np.floor(Dct2)
        SeqBlock = ZigZag(Coeff)
        Seq = Reduce(SeqBlock, QNew)
        iSeqBlock = Expand(Seq)
        iDct1 = iZigZag(iSeqBlock)
        iDct2 = sf.idct(iDct1, axis  = 1, norm = 'ortho')
        y = sf.idct(iDct2, axis  = 0, norm = 'ortho')
        iImage[ (i*8) : ((i*8)+8)  , (j*8) : ((j*8)+8)] = y
        
        
print(iImage)


print("iImage")
iImage = iImage + 127
print(iImage)
cv2.imwrite('Result.jpg',iImage)  


print("hello")
'''
cv2.imshow('1', cv2.imread("LENNA.JPG"))
cv2.imshow('2', cv2.imread("Result.jpg"))
y = input("do you want to kill all the windows?")
#cv2.waitKey()
cv2.destroyAllWindows() '''



"""Sample"""
