warn     = -Wall
debug    = -g
CPP      = g++ $(debug) $(warn)
progname = cluster2
libs     = -lm
INC      = cluster.h myrand.h



cluster: main.o cluster.o data.o sobol.o
	$(CPP) -o $(progname) main.o cluster.o data.o sobol.o $(libs)

test:	test.C cluster.o sobol.o sobol.h cluster.h interface.o
	$(CPP) -c test.C
	$(CPP) -o test test.o cluster.o sobol.o interface.o $(libs)

main.o: main.C $(INC)
	$(CPP) -c main.C

cluster.o: cluster.C $(INC) sobol.h
	$(CPP) -c cluster.C

data.o: data.C $(INC)
	$(CPP) -c data.C

sobol.o: sobol.C sobol.h
	$(CPP) -c sobol.C

interface.o: interface.C sobol.h cluster.h
	$(CPP) -c interface.C

all:
	g++ -c *.C
	g++ -o $(progname) *.o $(libs)

clean: 
	rm $(progname) *.o *~

pack:
	tar czvf ../cluster.tgz *.C *.h Makefile testdata times.pl

install: cluster
	cp /Users/ssabas/Desktop/mres-project/code/cpp/cluster2/


