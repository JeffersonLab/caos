# caos (Clas12 Online Software)

## Java Utilities

### Monitoring

The automated program can monitor the et ring by reading events from the stream and printing then at given intervals.
To compile and run the monitoring use:

```
prompt> mvn install
prompt> java -jar  target/data-0.8-SNAPSHOT-jar-with-dependencies.jar /tmp/et_hipo_test 3000
```

Will retrieve events from local ET ring with file "/tmp/et_hipo_test" every 3 seconds (time is given in msec).

### Manual Data monitoring

To run Java utilities use the following commands:


```
prompt> jshell --class-path target/data-0.8-SNAPSHOT-jar-with-dependencies.jar
jshell> import org.jlab.data.h5.*;
jshell> import j4np..h5.*;
jshell> DataSourceEt et = new DataSourceEt("localhost");

```
