from fractions import Fraction as frac
from operator import add
from operator import neg
import random
import numpy as np
from multiprocessing import Pool
from time import sleep
import inverse as inverse
import time
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

def egcd(a, b):
    # x,y, u,v = 0,1, 1,0
    # while a != 0:
    #     q, r = b//a, b % a
    #     m, n = x-u * q, y - v * q
    #     b,a, x,y, u,v = a,r, u,v, m,n
    # gcdVal = b
    # return gcdVal, x, y 
    if a == 0:
        return b, 0, 1
    else:
        gcdVal, x, y = egcd(b % a, a)
        return gcdVal, y - (b // a) * x, x

#Modular inverse
#An application of extended GCD algorithm to finding modular inverses:
def modinv(a, m):
    gcdVal, x, y = egcd(a, m)
    if gcdVal != 1:
        # print("error")
        return None  # modular inverse does not exist
    else:
        return x % m

#Modulus Function which handles Fractions as well
def fracMod(f,m):
    [tmp,t1,t2]=egcd(f.denominator,m)
    if tmp!=1:
      print("ERROR GCD of denominator and m is not 1")
      return None
      # 1/0
    else:
      out=modinv(f.denominator,m)*f.numerator % m
      return out

def trim(seq):
  if len(seq) == 0:
    return seq
  else:
    for i in range(len(seq) - 1, -1, -1):
      if seq[i] != 0:
        break
  return seq[0:i+1]

def resize(c1,c2):
  if(len(c1)>len(c2)):
    c2=c2+[0]*(len(c1)-len(c2))
  if(len(c1)<len(c2)):
    c1=c1+[0]*(len(c2)-len(c1))
  return [c1,c2]

def subPoly(c1,c2):
  [c1,c2]=resize(c1,c2) 
  c2=list(map(neg,c2))
  out=list(map(add, c1, c2))
  # print(type(out))
  return trim(out)

def modPoly(c,k):
  if(k==0):
    print("Error in modPoly(c,k). Integer k must be non-zero")
  else:
    return [fracMod(x,k) for x in c]

# def multPoly(c1,c2):
#   print("c1:",c1)
#   print("c2:",c2)
#   [c1,c2] = inverse.resize(c1,c2)
#   order=(len(c1)-1+len(c2)-1)
#   out=[0]*(order+1)
#   for i in range(0,len(c1)):
#     for j in range(0,len(c2)):
#       out[j+i]=out[j+i]+c1[i]*c2[j]
#   print("out:",trim(out))
#   return trim(out)

def multPoly(c1, c2):
    c1 = np.array(c1, dtype=object)
    c2 = np.array(c2, dtype=object)
    n = len(c1) + len(c2) - 1
    result = np.zeros(n, dtype=object)
    for i, c1_coeff in enumerate(c1):
        result[i:i+len(c2)] += c1_coeff * c2
    return trim(result).tolist()

def reModulo(num,div,modby):
  [_,remain]=divPoly(num,div)
  return modPoly(remain,modby)

def divPoly(N,D):
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

def parallel(S,T,Q,x):
   
   S[x]=subPoly(S[x-2],multPoly(Q[x-2],S[x-1]))
   T[x]=subPoly(T[x-2],multPoly(Q[x-2],T[x-1]))
   return


def extEuclidPoly(a,b):
  
    print("inside extEuclid")
    switch = False
    a=trim(a)
    b=trim(b)
    if len(a)<=len(b):
      a1, b1 = a, b
    else:
      a1, b1 = b, a
      switch = True
    Q,R=[],[]
    while b1 != [0]:
      [q,r]=divPoly(a1,b1)
      Q.append(q)
      R.append(r)
      a1=b1
      b1=r
    S=[[0]]*(len(Q)+2)
    T=[[0]]*(len(Q)+2)

  

    S[0],S[1],T[0],T[1] = [1],[0],[0],[1]
    # S = np.array(S, dtype=object)
    # T = np.array(T, dtype=object)

    # print("type : ",type(Q), type(S),type(T))
    # with Pool() as pool:
    #   pool.starmap(parallel,[[S,T,Q,x] for x in range(2,len(S))])
    # print("type : ",type(Q), type(S),type(T))
    # Parallel(n_jobs=2)(delayed(parallel)(S,T,Q,x) for x in range(2,len(S)))
    for x in range(2, len(S)):
        S[x]=subPoly(S[x-2],multPoly(Q[x-2],S[x-1]))
        T[x]=subPoly(T[x-2],multPoly(Q[x-2],T[x-1]))
        
    

    # jobs = []
    # for i in range(2, len(S)):
    #     jobs.append(delayed(parallel)(S, T, Q, i))
    # Parallel(n_jobs=2)(jobs)
      # S[x]=subPoly(S[x-2],multPoly(Q[x-2],S[x-1]))
      # T[x]=subPoly(T[x-2],multPoly(Q[x-2],T[x-1]))
      
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #   future_results = []
    #   for x in range(2, len(S)):
    #       future = executor.submit(parallel, S, T, Q, x)
    #       future_results.append(future)

    #   # Wait for all futures to complete
    #   concurrent.futures.wait(future_results)



    gcdVal=R[len(R)-2]
    s_out=S[len(S)-2]
    t_out=T[len(T)-2]
    ### ADDITIONAL STEPS TO SCALE GCD SUCH THAT LEADING TERM AS COEF OF 1:
    scaleFactor=gcdVal[len(gcdVal)-1]
    gcdVal=[x/scaleFactor for x in gcdVal]
    s_out=[x/scaleFactor for x in s_out]
    t_out=[x/scaleFactor for x in t_out]
    
    if switch:
      return [gcdVal,t_out,s_out]
    else:
      return [gcdVal,s_out,t_out]


# f = []
# for i in range(0,166):
#      n = random.randint(-1, 2)
#      f.append(n)
# f = [-1, 0, 0, 0, 1, -1, 1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
# q=7
# N=len(f)

# D=[0]*(N+1)
# D[0]=-1
# D[N]=1
# print ("D: ",D)

# [gcd_f,s_f,t_f]=extEuclidPoly(f,D)

# s_f=modPoly(s_f,q)
# print ("f=",f)
# print ("\ns_f=",s_f)


