# caos
Clas12 Online Software

# Running online AI/ML reconstruction

login into clonfarm11 computer with clasrun account. Then run script:

```
prompt> ~/CAOS/bin/start_hipo_et.sh
```

This will bring two terminals (xterm) one of them running local ET ring, the other
one running the ET to ET trnasfer decoder program, with some information printed
routinely about the transfer rates.

Once these terminals are running, open another terminal log in into clonfarm11
as clasrun. then from prompt run:

```
prompt>clas12mltrack
```

sit back and enjoy tracks reconstructed by artificial intelligence.

# Starting ET ring

on clon machines

```
et_start -n 100 -s 100000 -f /tmp/et_sys_decoder -p 12345
evio2et clasrun_005442.evio /tmp/et_sys_decoder
et_monitor -f /tmp/et_sys_decoder
```
