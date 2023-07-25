//******************************************************************
//*  ██╗  ██╗██╗██████╗  ██████╗     ██╗  ██╗    ██████╗
//*  ██║  ██║██║██╔══██╗██╔═══██╗    ██║  ██║   ██╔═████╗
//*  ███████║██║██████╔╝██║   ██║    ███████║   ██║██╔██║
//*  ██╔══██║██║██╔═══╝ ██║   ██║    ╚════██║   ████╔╝██║
//*  ██║  ██║██║██║     ╚██████╔╝         ██║██╗╚██████╔╝
//*  ╚═╝  ╚═╝╚═╝╚═╝      ╚═════╝          ╚═╝╚═╝ ╚═════╝
//************************ Jefferson National Lab (2023) ***********
//******************************************************************
//* Example program for writing HIPO-4 Files..
//* Includes defining schemas, opening a file with dictionary
//*-----------------------------------------------------------------
//* Author: G.Gavalian
//* Date:   02/03/2023
//******************************************************************

#include <iostream>
#include <stdlib.h>
#include "table.h"
#include "utils.h"
#include "bank.h"
#include "decoder.h"
#include "reader.h"
#include "writer.h"
#include "detectors.h"
#include "eviostream.h"

#define MAXBUF 10000000
#define ONE_MB 1048576
#define ONE_KB 1024

void benchmark_translate(std::vector<evio::container> &events, int iter);

int main(int argc, char** argv){

    printf("--->>> example program for data translation\n");
    char inputFile[256];

   if(argc>1) {
      sprintf(inputFile,"%s",argv[1]);
      //sprintf(outputFile,"%s",argv[2]);
   } else {
      std::cout << " *** please provide a file name..." << std::endl;
     exit(0);
   }
    
   std::vector<evio::container> events = evio::container::create(5000,100*1024);
   printf(">>> events container created with size = %d\n", (int) events.size());

   evio::eviostream stream;
   stream.open(inputFile);   
   bool success = stream.pull(events);  
   benchmark_translate(events,50);
}

/*=================================================================================
* Benchmarking decoding
===================================================================================*/

void benchmark_translate(std::vector<evio::container> &events, int iter){
    
    printf("\n\n ---- decoder benchmark ----\n");
    printf(" ---- # of events %d\n",(int) events.size());
    printf(" ---- iterations %d\n",iter);
    int counter = 0;
    int resetcounter = 0;
    hipo::event  eout(2*ONE_MB);
    hipo::benchmark   bench;
    coda::decoder     decoder;
    coda::eviodata_t  ptr;
    hipo::dataframe frame(100,8*ONE_MB);

    decoder.initialize();
    //decoder.showKeys();

    printf("\n\nstart:\n");
    bench.resume();
    for(int i = 0; i < iter; i++){
        for(int e = 0; e < events.size(); e++){
            decoder.reset();
            /*
            events[e].init();
            while(events[e].next() == true){
                events[e].link(ptr);
                if(ptr.offset>0){
                    //printf("crate (%3d) -> tag = %8X (%8d)\n",ptr.crate,ptr.tag,ptr.tag);
                    decoder.decode(ptr);
                }
                
                //decoder.decode(evioptr);
            }
            */
           decoder.decode(&events[e].getBuffer()[0]);
            if(events[e].size()>0) counter++;
            decoder.write(eout);
            bool status = frame.addEvent(eout);
            if(status == false){
                frame.reset(); frame.addEvent(eout);
                resetcounter++;
            }
            //printf("============================================================\n");
            //decoder.show();
            //counter++;
        }
        printf(".");fflush(stdout);if((i+1)%20==0) printf("  >  %d\n",i+1);
    }

    bench.pause();
    double time = bench.getTimeSec();
    double rate = ((double) counter)/time/1024.0;
    printf("\n\n");
    printf("-----------------------------------\n");
    printf("    >>>> benchmark results <<<< \n");
    printf("-----------------------------------\n");
    printf("::: events decoded : %d  \n",counter);
    printf(":::    frame reset : %d  \n",resetcounter);
    printf(":::   time elapsed : %.4f sec \n", time);
    printf(":::  decoding rate : %.4f kHz \n",rate);
    printf("-----------------------------------\n");
    printf("\n\ndone ....\n\n");
}