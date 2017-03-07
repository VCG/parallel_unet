import sys


def main(argv=None):
    if len(argv) < 3:
        print "usage getnunet_shape <depth> <width> <size>"
        return

    depth = int(argv[0])
    width = int(argv[1])
    size  = int(argv[2])

    sample = 2

    # [3, 116, 116]
    sampling_range = depth*2
    for d in range(sampling_range+1):

        for w in range(width-1):
            print '[', size, size, ']'
            size -= sample
        print '[', size, size, ']'


        if d < (sampling_range/2):
            size /= 2
        else:
            size *=2
        

if __name__ == "__main__":
    main(sys.argv[1:])
