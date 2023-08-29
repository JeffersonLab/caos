#!/bin/sh
export LEVEL3DIR=`dirname $0`/..
OPTIONS="-Dsun.java2d.pmoffscreen=false -Xmx8048m -Xms2048m"

maxCores=$1
for i in `seq 1 $maxCores`
do

    export OMP_NUM_THREADS=`expr $i \* 2`
    echo $i $OMP_NUM_THREADS
    java -cp "$LEVEL3DIR/target/*" org.jlab.online.level3.Level3Processor -i 256 -batch 512 $2
done
