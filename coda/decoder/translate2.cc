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
    
   std::vector<evio::container> events = evio::container::create(50,100*1024);
   printf(">>> events container created with size = %d\n", (int) events.size());

    hipo::event  eout(2*ONE_MB);
    hipo::benchmark   bench;
    coda::decoder     decoder;
    coda::eviodata_t  ptr;

    decoder.initialize();

   evio::eviostream stream;
   stream.open(inputFile);
   int   counter = 0;
   bool  success = true;
   while(success==true){
        success = stream.pull(events);  counter++;
        //events[0].init();
        for(int k = 0; k < events.size(); k++){
            //printf(">>> parsing even # %d\n",k);
            if(events[k].size()>0){
                events[k].init();
                while(events[k].next()==true){
                    events[k].link(ptr);
                    if(ptr.offset>0){
                    //printf("crate (%3d) -> tag = %8X (%8d)\n",ptr.crate,ptr.tag,ptr.tag);
                    decoder.decode(ptr);
                    }
                }
                //decoder.show();
                decoder.write(eout,events[k]);
                stream.push(eout);
            }
        }
    }
    stream.close();
    printf("events read = %d\n",counter);

}
