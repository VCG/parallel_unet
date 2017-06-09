import sys


def main(argv=None):
    if len(argv) < 3:
        print "usage getnunet_shape <depth> <width> <size>"
        return

    # e.g. python genunet_shape.py 4 3 496  (4 is depth, 3 is width, 496 is input size)

    depth = int(argv[0]) # the depth of the network - number of blocks before upsampling
    width = int(argv[1]) #
    size  = int(argv[2])

    sample = 2

    input2d  = [size, size]
    input3d  = [size, size, size]
    output2d = []
    output3d = []

    # [3, 116, 116]
    sampling_range = depth*2
    for d in range(sampling_range+1):

        for w in range(width-1):
            print '[', size, size, ']'
            size -= sample
        print '[', size, size, ']'

        if d == sampling_range:
            output2d = [size, size]
            output3d = [size, size, size]

        if d < (sampling_range/2):
            size /= 2
        else:
            size *=2

    s  = 'netconf.input_shape = ['
    o  = 'netconf.output_shape = ['
    for i in range(len(input2d)):
        if i == 0:
            s = '%s%d'%(s, input2d[i])
            o = '%s%d'%(o, output2d[i])
        else:
            s = '%s,%d'%(s, input2d[i])
            o = '%s,%d'%(o, output2d[i])
    s += ']'
    o += ']'
    print s
    print o

    s  = 'netconf.input_shape3d = ['
    o  = 'netconf.output_shape3d = ['
    for i in range(len(input3d)):
        if i == 0:
            s = '%s%d'%(s, input3d[i])
            o = '%s%d'%(o, output3d[i])
        else:
            s = '%s,%d'%(s, input3d[i])
            o = '%s,%d'%(o, output3d[i])
    s += ']'
    o += ']'
    print s
    print o


    

if __name__ == "__main__":
    main(sys.argv[1:])
