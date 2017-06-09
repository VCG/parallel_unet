import os

for i in range(0, 1000, 100):
    model = 'net_iter_%d.caffemodel'%(i)
    if os.path.exists( model ):
        os.remove( model )
        os.remove('net_iter_%d.solverstate'%(i))
