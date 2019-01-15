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

vals = {}
def add(k,v):
    if not k in vals:
        vals[k] = []
    vals[k].append(float(v))

print "Method,MSE R, DEV R, MSE G, DEV G, MSE B, DEV B"
for fn in argv[1:]:
    vals = {}
    f=open(fn)
    for l in f.readlines():
        parts = re.split(',\s*',l)
        mr = parts[3]; mg = parts[4]; mb = parts[5];
        add('R',mr); add('B',mb); add('G',mg);

    print fn+",",
    for c in 'RGB':
        average,d = av(vals[c])
        errp,errn = errors(vals[c])
        #print "%02.2f, %02.2f   %02.2f, %02.2f,  " %(average,d,errp,errn),
        print "%02.2f, %02.2f,   " %(average,d),
        #print average,
    #print (average,errp,errn)
    print
    
