#!/bin/sh
export CAOSDIR=`dirname $0`/..
OPTIONS="-Dsun.java2d.pmoffscreen=false -Xmx2048m -Xms1024m"
jshell --class-path "$CAOSDIR/java/data/target/*" --startup $CAOSDIR/etc/imports.jshell $*
