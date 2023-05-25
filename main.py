import inverse as inverse
import rand as rand
import polArithmetic as polArithmetic
import time


def key_gen(df, dg, deg, q, p):
    f = rand.randpol(df, df - 1, deg + 1)
    g = rand.randpol(dg, dg, deg + 1)
    print("f is:", f)
    print("g is:", g)
    N = len(f)
    D = [0]*(N+1)
    D[0] = -1
    D[N] = 1
    [gcd_f, s_f, t_f] = inverse.extEuclidPoly(f, D)
    finvp = inverse.modPoly(s_f, p)
    finvq = inverse.modPoly(s_f, q)
    # print("finvp is:",finvp)
    # print("finvq is:",finvq)
    while any(x is None for x in finvp) or any(x is None for x in finvq) or finvp is None or finvq is None or all(x == 0 for x in finvp) or all(x == 0 for x in finvq):
        # print("Inverse does not exist")
        # return [None,None]
        print("inside while")
        f = rand.randpol(df, df - 1, deg + 1)
        [gcd_f, s_f, t_f] = inverse.extEuclidPoly(f, D)
        finvp = inverse.modPoly(s_f, p)
        finvq = inverse.modPoly(s_f, q)
    print("finvp is:", finvp)
    print("finvq is:", finvq)
    for i in range(len(g)):
        g[i] = g[i]*p
    # print("g changed is:",g)
    # if(len(g) == len(finvq)):
    #     print("same size")
    h = polArithmetic.star_multiply(g, finvq, q)
    # print("h is:",h)
    pk = [f, finvp]
    return h, pk

# h is the public key
# m - message


def encrypt(h, dr, msg, q, deg):
    r = rand.randpol(dr, dr, deg + 1)
    temp = polArithmetic.star_multiply(h, r, q)
    e = polArithmetic.polAdd(msg, temp, q)
    return e


def decrypt(pk, e, p, q):
    a = polArithmetic.star_multiply(e, pk[0], q)
    r = q/2
    for i in range(len(a)):
        if a[i] < -1*r:
            a[i] = a[i] + q
        elif a[i] > r:
            a[i] = a[i] - q
    for i in range(len(a)):
        a[i] = a[i] % p
    c = polArithmetic.star_multiply(a, pk[1], p)
    return c


if __name__ == '__main__':
    N = int(input("Enter N:"))
    p = int(input("Enter p:"))
    q = int(input("Enter q:"))
    dr = int(input("Enter dr:"))
    df = int(input("Enter df:"))
    dg = int(input("Enter dg:"))

    # start time
    start = time.time()

    [pub, priv] = key_gen(df, dg, N - 1, q, p)
    if(pub == None):
        exit()

    # end time
    end = time.time()
    print("elapsed time is:", end - start, "seconds")
    # print("pub is:",pub)
    # print("priv is:",priv)
    # msg = [0,1,0,1,1]
    n = int(input("Enter the length of message:"))
    msg = []
    for i in range(n):
        msg.append(int(input()))
    ciphertext = encrypt(pub, dr, msg, q, N - 1)
    print("ciphertext:", ciphertext)
    decrypt = decrypt(priv, ciphertext, p, q)[0:len(msg)]
    # change 2 to -1 in msg by iterating thrugh it
    print("decrypted:", decrypt)
    # blocks = []
    # ciphertext = []
    # if(len(msg)<N):
    #     inverse.resize(msg,pub)
    #     blocks.append(msg)
    # else:
    #     for i in range(0,len(msg),N):
    #         if(i+N <= len(msg)):
    #             blocks.append(msg[i:i+N])
    #     if(len(msg) % N):
    #         inverse.resize(msg[i:len(msg)],pub)
    #         blocks.append(msg[i:i+N])

    # for i in range (0,len(blocks)):
    #     ciphertext.append(encrypt(pub,dr,blocks[i],q,N - 1))
    #     print("Original message: ",blocks[i])
    #     print("ciphertext[" ,i,"] : ",ciphertext[i])
    #     if(len(blocks[i]) < len(ciphertext[i])):
    #         print("Decrypted message: ",decrypt(priv,ciphertext[i],p,q)[0:len(blocks[i])])
    #     else:
    #         print("Decrypted message: ",decrypt(priv,ciphertext[i],p,q))

    # intel i7/i9 processors recommended

    # recorded times
    # N = 31,p=3,q=32 :- time = 0.09404826164245605 seconds
    # N = 41,p=3,q=32 :- time = 0.33559703826904297 seconds
    # N = 51,p=3,q=32 :- time = 0.9701879024505615 seconds
    # N = 61,p=3,q=32 :- time = 4.295067071914673 seconds
    # N = 71,p=3,q=32 :- time = 9.976176261901855 seconds
    # N = 81,p=3,q=32 :- time = 18.92745542526245 seconds
    # N = 91,p=3,q=32 :- time = 60.08960580825806 seconds
    # N = 101,p=3,q=32 :- time = 102.63305878639221 seconds
