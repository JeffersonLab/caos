# caos
Clas12 Online Software

# Running online reconstruction (Conventional or AI/ML) 

login into clonfarm11 computer with clasrun account. Then run script:

```
clonfarm11> ~/CAOS/bin/start_hipo_et.sh
```

This will bring two terminals (xterm) one of them running local ET ring, the other
one running the ET to ET trnasfer decoder program, with some information printed
routinely about the transfer rates.

Once these terminals are running, open another terminal log in into clonfarm11
as clasrun. then from prompt run:

```
clonfarm11> clas12online
```
This will start the online conventional reconstruction.Enjoy the histograms.

# AI/ML reconstruction

Once these terminals are running, open another terminal log in into clonfarm11
as clasrun. then from prompt run:

```
clonfarm11> clas12mltrack
```
# Expert Only (for debugging)

This section is for debugging the online ET to ET connection on clon machines.
Do not run these commands during run taking, it may cause DAQ et ring to crash.
For shift takers run commands from previous section.

To create an ET ring from a file use the following commands: (run this
on clondaq7).

```
clondaq7> et_start -n 500 -s 100000 -f /tmp/et_sys_decoder
clondaq7> evio2et clasrun_005442.evio /tmp/et_sys_decoder
clondaq7> et_monitor -f /tmp/et_sys_decoder
```

Then a ring can be created on clonfarm11, and populated by HIPO data frames 
using commands:

```
prompt> et_start -n 500 -s 100000 -f /tmp/et_sys_decoder
prompt> cd /home/clasrun/CAOS 
prompt> /usr/clas12/release/1.4.0/coda/src/hipo4/Linux_x86_64/bin/et2hipo2et clondaq7:/tmp/et_sys_decoder clonfarm11:/tmp/et_sys_decoder ET2ET
```



