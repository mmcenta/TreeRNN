direc='data/reutersnews/'
    with open('reuters_target.txt','w+') as fd_target:
        with open('reuters.txt','w+') as fd:
            with open(direc+'topics_train.txt','r') as inp:
                for l in inp.readlines():
                    l=l.split('\t')
                    fd_target.write(l[0]+'\n')
                    fd.write(l[1]+'\n')
            with open(direc+'topics_test.txt','r') as inp:
                for l in inp.readlines():
                    l=l.split('\t')
                    fd.write(l[0]+'\n')
                    fd.write(l[1]+'\n')