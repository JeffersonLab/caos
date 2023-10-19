#!/bin/sh
export LEVEL3DIR=`dirname $0`/..
OPTIONS="-Dsun.java2d.pmoffscreen=false -Xmx8048m -Xms2048m"
maxCores=$1
export OMP_NUM_THREADS=6
echo $i $OMP_NUM_THREADS
java -cp "$LEVEL3DIR/target/*" org.jlab.online.level3.Level3ProcessFile $*
