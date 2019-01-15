#!/usr/bin/python
from math import *
from sys import argv
import re
import sys
from sys import stderr
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
def join(s,l):
    return s.join(map(lambda x:str(x), l))

vals = {}
def put(f,p,t):
    if not p in vals:
        vals[p] = {}
    fs = vals[p]
    if not f in fs:
        fs[f] = []
    fs[f].append(t)

DEV=False
if '-std' in argv:
    DEV=True
    argv.remove('-std')

title = "No. pixel, "
for f in argv[1:]:
    fh = open(f);
    fh.readline()
    for l in fh.readlines():
        #(func,file,x,y,pixels,rt,ut,st,mask,total) = re.split(',\s*/',l)
        l = re.split(',\s*',l);
        #print l
        
        ut = l[6]; st = l[7]; pixels = l[4]
        total = float(ut) + float(st);
        put(f,pixels,total)
    if DEV:
        title += "%s Average Time, %s StdDev, " %(f,f)
    else: 
        title += "%s Average Time, " %(f)
        
print title
for pixels in vals.keys():
    res = []
    for f in sys.argv[1:]:
        #for f in vals[pixels].keys():
        if f in vals[pixels]:
           #stderr.write("%s %s pixels %i samples\n"%(f[:5],pixels,len(vals[pixels][f])))
           times = vals[pixels][f];
           (a,dev) = av(times)
           #print pixels,f,times;
        else:
	    a,dev = "NA","NA" 
        stderr.write("%s av:%f dev:%f\n"%(f,a,dev));
        res.append(a)
        if DEV:
            res.append(dev)
    print str(pixels)+','+ join(",",res)


