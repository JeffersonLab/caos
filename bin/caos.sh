#!/bin/sh
export CAOSDIR=`dirname $0`/..
OPTIONS="-Dsun.java2d.pmoffscreen=false -Xmx2048m -Xms1024m"
jshell --class-path "$CAOSDIR/java/data/target/data-0.8-SNAPSHOT-jar-with-dependencies.jar" --startup $CAOSDIR/etc/imports.jshell $*
