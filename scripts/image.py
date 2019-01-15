#!/usr/bin/env python
import Image
import sys,collections

ACTIONS = "convert bayer split red green blue compare diff".split(" ")

def usage():
    print "Usage: %s action [params] image.ppm result.ppm" % sys.argv[0]
    print "Actions: %s" % " ".join(ACTIONS)
    sys.exit(1)

RGB = 3
R = 0
G = 1
B = 2

def get_filter_color(x,y):
    if y&1:
        if x&1:
            return B
        else:
            return G
    else:
        if x&1:
            return G
        else:
            return R


_chans = {"red":0,"green":1,"blue":2}
def get_channel_num(chan):
    if chan in _chans:
        return _chans[chan]
    return -1

_names = {0:"red",1:"green",2:"blue"}
def get_channel_name(n):
    if n in _names:
        return _names[n]
    return `n`    

def bayer(size,data):
    width = size[0]
    height = size[1]
    data = list(data)
    for y in range(height):
        for x in range(width):
            i = y*width + x
            pix = list(data[i])
            fc = get_filter_color(x,y)
            for c in range(RGB):
                if fc != c:
                    pix[c] = 0
            data[i] = tuple(pix)
    return data
                

    
def eval_RGB(image,func):
    width = image.size[0]
    height = image.size[1]
    data = list(image.getdata())
    for y in range(height):
        for x in range(width):
            i = y*width + x
            pix = data[i]
            res = func(pix)
            data[i] = res
    image.putdata(data)


def channel(chan,pix):
    res = list(pix)
    for c in range(RGB):
        if chan != c:
            res[c] = 0
    return tuple(res)

def diff(chan,pix):
    if (chan >= 0):
        res = abs(pix[chan] - pix[G])
        return (res,res,res)
    else:
        diffR = abs(pix[R] - pix[G])
        diffB = abs(pix[B] - pix[G])
        return (diffR,0,diffB)

def pc(n):
    return "%0.2f%%" % n
    
def compare(im,im2):
    if (im.size != im2.size):
        print "Images are different sizes: %s %s" %(`im.size`, `im2.size`)
        return 1
    data = im.getdata()
    data2 = im2.getdata()

    width,height = im.size
    sq_error = [0,0,0]
    ndiffs = [0,0,0]
    
    for y in range(height):
        for x in range(width):
            i = y*width + x            
            for c in range(RGB):
                #if channel == -1 or channel == c:
                diff = data[i][c] - data2[i][c]
                sq_error[c] += diff**2
                if diff != 0:
                    ndiffs[c] += 1
                    
    if ndiffs == [0,0,0]:
        print "Images are identical"
        return

    n = width * height
    mean_sq_error = float(sq_error[R] + sq_error[G] + sq_error[B]) / (n * RGB)
    mean_sq_err_chan = [0,0,0]
    for c in range(RGB):
        mean_sq_err_chan[c] = float(sq_error[c]) / n
    
    print "Total mean square error %s" % pc(mean_sq_error)
    print "Green channel MSE %s" % pc(mean_sq_err_chan[B])
    print "Red   channel MSE %s" % pc(mean_sq_err_chan[R])
    print "Blue  channel MSE %s" % pc(mean_sq_err_chan[G])
    
def pairs_to_dict(l):
    d = collections.defaultdict(bool)
    for p in l:
        if "=" in p:
            key,val = p.split("=",1)
            d[key] = val
        else:
            d[p] = True            
    return d

if len(sys.argv) < 4:
    usage()

action = sys.argv[1]
src,result = sys.argv[-2:]
params = pairs_to_dict(sys.argv[2:-2])

if not action in ACTIONS:
    usage()

im = Image.open(src)
#print im.mode


if action == "bayer":
    print "Making bayer image from %s" % src    
    res = Image.new(im.mode,im.size)
    res.putdata(bayer(im.size,im.getdata()))
elif action == "convert":
    res = im
elif action == "split":
    print "Extracting RGB channels"
    src = im.getdata()
    size = im.size
    for chan in range(RGB):
        res = im.copy()
        eval_RGB(res,lambda pix: channel(chan,pix))
        res.save(result+get_channel_name(chan)+".ppm")
    sys.exit(0)
        
    
elif action in ["red","green","blue"]:
    print "Extracting %s channel" %action
    chan = get_channel_num(action)
    eval_RGB(im,lambda pix: channel(chan,pix))
    res = im

elif action == "compare":    
    print "Comparing %s and %s" %(src,result)
    
    im2 = Image.open(result)
    compare(im,im2)
    sys.exit(0) # do not save image

elif action == "diff":
    channel = params["channel"]
    print "Difference between green and %s channels" % (channel if channel else "blue and red")
    chan = get_channel_num(channel)
    eval_RGB(im,lambda pix: diff(chan,pix))
    res = im

    



else:
    usage()

print "Saving result to %s" % result
res.save(result)
