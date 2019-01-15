#!/usr/bin/env python
import Image
import sys,collections
from colortransforms import *

from math import sqrt
def abdist(t1,t2):
    (l1,a1,b1) = t1
    (l2,a2,b2) = t2
    return sqrt((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2))

from time import *
import sys
def zipper(i1,i2,nam,c1):
    
    im2 = Image.open(i2)
#    assert(im1.size == im2.size)
    sys.stderr.write("Running "+i1 + " "+nam+"\n")
    S=0
    N=0
    (w,h) = im2.size
    
    d2 = im2.getdata()
    c2 = map(lambda t:rgb_to_cielab(*t),d2)
    
    for y in range(1,h-1):
        for x in range(1,w-1):
            vals = {}
            offset = y*w+x
            for dx in range(-1,2):
                for dy in range(-1,2):
                    if not (dx == 0 and dy == 0):
                        i = x+dx
                        j = y+dy
                        n = j*w+i
                        v1 = c1[n]
                        v2 = c1[offset]
                        #print v1,v2
                        d = abdist(v1,v2)
                        vals[d] = (i,j)
            minv = min(vals.keys())
            (x1,y1) = vals[minv]
            psrc1 = c1[offset]
            psrc2 = c2[offset]
            p1 =    c1[y1*w+x1]
            p2 =    c2[y1*w+x1]
            d1 = abdist(psrc1,p1)
            d2 = abdist(psrc2,p2)
            diff = abs(d2-d1)
            #print diff
            if diff >= 2.3:
                S += diff
                N += 1
            #print diff
    p = (S/(h*w))*100
    
    date = strftime("%d-%b-%Y %H:%M:%S",localtime())
    np = (float(N)/(h*w))*100
    print "%s, %s, %s, %0.2f, %0.2f" %(date,i1,nam,p,np)
        
P="/home/sjf/e/sample_images/quality/"
lines = open(sys.argv[1]).readlines()
F=[(".bilin.ppm","bil"),(".ahd.ppm","AHD"),(".ahdmask.ppm","Mask"),(".ahdmask2.ppm","mask2")]
for l in lines:
    sys.stderr.write(l)
    try:
        l = l.strip()
        f = P+l
        
        im1 = Image.open(f+".orig.ppm")
        
        d1 = im1.getdata()
        c1 = map(lambda t:rgb_to_cielab(*t),d1)
        
        for ext in F:
            zipper(l,f+ext[0],ext[1],c1)
    except Exception,e:
        print e


## action = sys.argv[1]
## src,result = sys.argv[-2:]
## params = pairs_to_dict(sys.argv[2:-2])

## if not action in ACTIONS:
##     usage()

## im = Image.open(src)
## #print im.mode


## if action == "bayer":
##     print "Making bayer image from %s" % src    
##     res = Image.new(im.mode,im.size)
##     res.putdata(bayer(im.size,im.getdata()))
## elif action == "convert":
##     res = im
## elif action == "split":
##     print "Extracting RGB channels"
##     src = im.getdata()
##     size = im.size
##     for chan in range(RGB):
##         res = im.copy()
##         eval_RGB(res,lambda pix: channel(chan,pix))
##         res.save(result+get_channel_name(chan)+".ppm")
##     sys.exit(0)
        
    
## elif action in ["red","green","blue"]:
##     print "Extracting %s channel" %action
##     chan = get_channel_num(action)
##     eval_RGB(im,lambda pix: channel(chan,pix))
##     res = im

## elif action == "compare":    
##     print "Comparing %s and %s" %(src,result)
    
##     im2 = Image.open(result)
##     compare(im,im2)
##     sys.exit(0) # do not save image

## elif action == "diff":
##     channel = params["channel"]
##     print "Difference between green and %s channels" % (channel if channel else "blue and red")
##     chan = get_channel_num(channel)
##     eval_RGB(im,lambda pix: diff(chan,pix))
##     res = im

    



## else:
##     usage()

## print "Saving result to %s" % result
## res.save(result)
