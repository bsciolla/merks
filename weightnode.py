

def weightnode(a):
    w = 2**(len(a)) - 1
    for (scale, i) in enumerate(a):
        w = w + (i)*(2**(len(a)-scale-1))
    return(w)
