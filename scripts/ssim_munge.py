#!/usr/bin/python
from sys import argv
import sys
from math import *
import re;
def fdiv(a,b):
    return float(a)/float(b)

def av(l):
    s = sum(l)
    n = len(l)
    av = fdiv(s,n);
    dev = 0;
    for i in l:
        dev += (av - i)**2;
    dev = sqrt(dev/n);
    return (av,dev);

def errors(l):
    a,d = av(l)
    p = 0.0;
    n = 0.0;
    #print
    for i in l:
        if i>a:
            p += (a-i)**2
        else:
            n += (a-i)**2
        #print '  ...',p,n

    p = sqrt(p/ len(l));
    n = sqrt(n/len(l))
    #print '  ',p,n
    return (p,n)

def join(s,l):
    return s.join(map(lambda x:str(x), l))

vals = []
def add(v):
    vals.append(float(v))
def clear():
    vals = []

print "Method,SSIM, SSIM dev"
for fn in argv[1:]:
    clear()
    f=open(fn)
    for l in f.readlines():
        parts = re.split(',\s*',l)
	ssim = parts[6]
        add(ssim); 

    print fn+",",
    average,d = av(vals)
    #errp,errn = errors(vals[c])
    #print "%02.2f, %02.2f   %02.2f, %02.2f,  " %(average,d,errp,errn),
    print "%02.4f, %02.4f,   " %(average,d),
    #print average,
    #print (average,errp,errn)
    print
    
