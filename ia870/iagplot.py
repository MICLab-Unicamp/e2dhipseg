# -*- encoding: utf-8 -*-
# Module iagplot

from numpy import *

def iagplot(plotitems=[], commands=[], ptitle="", xlabel="", ylabel="", size=(320,240)):


    import tempfile, shutil
    import adpil
    #
    tmpdir = tempfile.mkdtemp('', 'iagplot_')
    if not os.path.isdir(tmpdir):
        raise IOError('Cannot create temporary directory')
    #
    imgfilename = 'iagplot_IMG.png'
    cmdfilename = 'iagplot_CMD.cmd'
    dataprefix  = 'iagplot_DATA'
    #
    currdir = os.getcwd()
    os.chdir(tmpdir)
    #
    try:
        cmdf = open(cmdfilename, 'w')
        ver = get_gnuver()
        if ver.startswith('4.'):
            print('set terminal png size %d,%d transparent small' % size, file=cmdf)
        else:
            print('set terminal png transparent small picsize %d %d' % size, file=cmdf)
        print('set output "%s"' % imgfilename.replace('\\', '/'), file=cmdf)
        print('set title "%s"'  % ptitle, file=cmdf)
        print('set xlabel "%s"' % xlabel, file=cmdf)
        print('set ylabel "%s"' % ylabel, file=cmdf)
        for cmd in commands:
            print(cmd, file=sys.stderr)
        NN = len(plotitems)
        strlist = []
        for ii in range(NN):
            item = plotitems[ii]
            datafile = '%s_%d.dat' % (dataprefix, ii+1)
            dataf = open(datafile, 'w')
            style = ''
            N = len(item)
            if N == 1:
                yy = asarray(item[0]).flat
                xx = arange(len(yy))
            else:
                yy = asarray(item[1]).flat
                if item[0] is None or len(item[0]) != len(yy):
                    xx = arange(len(yy))
                else:
                    xx = asarray(item[0]).flat
            if N == 3:
                style = 'title "%s" with lines' % (item[2],)
            elif N == 4:
                style = 'title "%s" with %s' % (item[2], item[3])
            else:
                style = 'title "" with lines'

            for i in range(len(yy)):
                print('%f %f' % (xx[i], yy[i]), file=dataf)
            dataf.close();
            strlist.append('"%s" %s' % (datafile, style))

        print('plot', ', '.join(strlist), file=cmdf)
        cmdf.close()
        #
        os.system('"%s" %s' % (GNUPLOT, cmdfilename))
        img = adpil.adread(imgfilename)
        #
    finally:
        os.chdir(currdir)
        shutil.rmtree(tmpdir, True)

    return img

