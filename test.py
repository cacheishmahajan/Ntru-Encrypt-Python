import time
import random
from multiprocessing import Pool
import numpy as np
import math
import inverse as inverse
from fractions import Fraction as frac 
from operator import neg
from operator import add
import multiprocessing as mp


def add(index,pol1,pol2,mod):
    return (pol1[index] + pol2[index]) % mod


def subPoly(c1,c2):
  [c1,c2]= inverse.resize(c1,c2) 
  c2=list(map(neg,c2))
  out=list(map(add, c1, c2))
  # print(type(out))
  return trim(out)


def divPoly(N,D):
    [N,D] = inverse.resize(N,D)
    N, D = list(map(frac,trim(N))), list(map(frac,trim(D)))
    degN, degD = len(N)-1, len(D)-1
    if(degN>=degD):
      q=[0]*(degN-degD+1)
      while(degN>=degD and N!=[0]):
        d=list(D)
        [d.insert(0,frac(0,1)) for i in range(degN-degD)]
        q[degN-degD]=N[degN]/d[len(d)-1]
        d=[x*q[degN-degD] for x in d]
        N=subPoly(N,d)
        degN=len(N)-1
      r=N 
    else:
      q=[0]
      r=N
    return [trim(q),trim(r)]

def divPoly2(N,D):
    [N, D] = inverse.resize(N, D)
    N, D = list(map(frac, trim(N))), list(map(frac, trim(D)))
    degN, degD = len(N) - 1, len(D) - 1
    if degN >= degD:
        q = np.zeros(degN - degD + 1, dtype=object)
        while degN >= degD and N != [0]:
            d = np.zeros(len(D) + degN - degD, dtype=object)
            d[degN - degD:] = D
            q[degN - degD] = N[degN] / d[len(d) - 1]
            d = np.array([x * q[degN - degD] for x in d], dtype=object)
            N = subPoly(N, d)
            degN = len(N) - 1
        r = N
    else:
        q = [0]
        r = N
    return [trim(q), trim(r)]


def polAdd(pol1,pol2,mod):
    answer = []
    
    with Pool() as pool:
        answer = pool.starmap(add, [[i,pol1,pol2,mod] for i in range(0,len(pol1))])
    return answer

def trim(seq):
  if len(seq) == 0:
    return seq
  else:
    for i in range(len(seq) - 1, -1, -1):
      if seq[i] != 0:
        break
  return seq[0:i+1]

def mul(out,c1,c2,i):
   for j in range(0,len(c2)):
        out[j+i]=out[j+i]+c1[i]*c2[j]
   
   
def multPoly(c1,c2):
   order=(len(c1)-1+len(c2)-1)
   out=[0]*(order+1)
   with Pool() as pool:
       pool.starmap(mul, [[out,c1,c2,i] for i in range(0,len(c1))])
   return trim(out)

    
  

def multPoly2(c1,c2):
  order=(len(c1)-1+len(c2)-1)
  out=[0]*(order+1)
  for i in range(0,len(c1)):
    for j in range(0,len(c2)):
      out[j+i]=out[j+i]+c1[i]*c2[j]
  return trim(out)

def multPoly3(c1, c2):
  n = len(c1) + len(c2) - 1
  c1_pad = np.pad(c1, (0, n - len(c1)), 'constant')
  c2_pad = np.pad(c2, (0, n - len(c2)), 'constant')
  c1_fft = np.fft.fft(c1_pad)
  c2_fft = np.fft.fft(c2_pad)
  out_fft = c1_fft * c2_fft
  out = np.fft.ifft(out_fft).real
  for i in range (0,len(out)):
     out[i] = round(out[i])
  return trim(out).astype(int).tolist()

def multPoly4(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    n = len(c1) + len(c2) - 1
    result = np.zeros(n, dtype=c1.dtype)
    for i, c1_coeff in enumerate(c1):
        result[i:i+len(c2)] += c1_coeff * c2
    return trim(result).astype(int).tolist()

if __name__ == "__main__":
    pol1 = [1,2,6,3,5,9,12,6,1,2,6,3,5,9,12,6]
    pol2 = [5,8,3,65,7,4,89,12,1,2,6,3,5,9,12,6]

    for i in range (1,5020):
        pol1.append(random.randint(100,200) % 2)
        pol2.append(random.randint(1,100)% 2)

    pol = []
    start = time.time()
    poll = multPoly4(pol1,pol2)
    print("First pol:",poll)
    print(type(poll))
    end = time.time()
    time1 = end - start
    
    
    start = time.time()
    pol = multPoly2(pol1,pol2)
    print("second pol:",pol)
    print(type(pol))
    end = time.time()
    
    print(time1 , " seconds taken with optimisation\n")
    print(end - start , " seconds taken without optimisation\n")


    if(poll == pol):
      print("same")
    else:
      print("not same")

