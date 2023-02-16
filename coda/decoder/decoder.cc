//******************************************************************************
//*       ██╗  ██╗██╗██████╗  ██████╗     ██╗  ██╗    ██████╗                  *
//*       ██║  ██║██║██╔══██╗██╔═══██╗    ██║  ██║   ██╔═████╗                 *
//*       ███████║██║██████╔╝██║   ██║    ███████║   ██║██╔██║                 *
//*       ██╔══██║██║██╔═══╝ ██║   ██║    ╚════██║   ████╔╝██║                 *
//*       ██║  ██║██║██║     ╚██████╔╝         ██║██╗╚██████╔╝                 *
//*       ╚═╝  ╚═╝╚═╝╚═╝      ╚═════╝          ╚═╝╚═╝ ╚═════╝                  *
//************************ Jefferson National Lab (2017) ***********************
/*
 *   Copyright (c) 2017.  Jefferson Lab (JLab). All rights reserved. Permission
 *   to use, copy, modify, and distribute  this software and its documentation
 *   for educational, research, and not-for-profit purposes, without fee and
 *   without a signed licensing agreement.
 *
 *   IN NO EVENT SHALL JLAB BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL
 *   INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
 *   OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF JLAB HAS
 *   BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *   JLAB SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *   PURPOSE. THE HIPO DATA FORMAT SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 *   ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". JLAB HAS NO OBLIGATION TO
 *   PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 *   This software was developed under the United States Government license.
 *   For more information contact author at gavalian@jlab.org
 *   Department of Experimental Nuclear Physics, Jefferson Lab.
 */
/*******************************************************************************
 * File:   decoder.cc
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */


#include "decoder.h"
#include <fstream>

namespace coda {

    void decoder::print(int x)
    {
        printf("%5d",x);;
    }

    void decoder::initialize(){

        components.push_back(new coda::component_ec("ec_tt.txt","ec_fadc.txt"));
        components.push_back(new coda::component_dc("dc_tt.txt"));
        components.push_back(new coda::component_config());

         constructDetectorSet();
    }

    void   decoder::constructDetectorSet(){
        printf(">>> creating component keys\n");
        for(int loop = 0; loop < components.size(); loop++){
             std::set<int> keys = components[loop]->keys();
             componentSets.push_back(keys);
        }

        for(int loop = 0; loop < components.size(); loop++){
             components[loop]->init();
        }
        //    componentSets.push_back(components[loop]->keys());
        printf(">>> done... creating component keys\n");
        showKeys();
    }

    void   decoder::showStats(){
        printf(" decoder statistics: \n");
        printf("\tavergate event size : %.2f kb\n",eventStats[eventStats.size()-1]/1024.);
        printf("\t     events decoded : %d\n",statisticsCounter);
        printf("\t      decoding time : %.2f sec\n", bench.getTimeSec());
        printf("\t      decoding rate : %d Hz\n", (int) ((statsPrintFrequency) / bench.getTimeSec()));

        for (int j = 0; j <=  4; j++) std::cout << "\033[A\033[2K";
    }

    void   decoder::plotStats(){
        if(eventStats.size()>3){
            ascii::Asciichart asciichart({ {"EVENTSIZE",eventStats} });
            std::cout << asciichart.show_legend(true).height(statisticsHeight).Plot();
            for (int j = 0; j <=  statisticsHeight; j++) std::cout << "\033[A\033[2K";
        }
    }

    void  decoder::showKeys(){
        for(int loop = 0; loop < componentSets.size(); loop++){
            printf(" component [%9s] keys : ",components[loop]->getName().c_str());
            std::set<int>::iterator it;
            for(it=componentSets[loop].begin(); it!=componentSets[loop].end(); ++it) printf("%3d ",*it);
            //for_each(componentSets[loop].begin(), componentSets[loop].end(),decoder::print);
            printf("\n");
        }
    }

    void decoder::decode(coda::eviodata_t &evio){
        for(int k = 0; k < componentSets.size(); k++){
            if(componentSets[k].count(evio.crate)>0){
                components[k]->decode(evio);
                break;
            }
        }
    }

    void decoder::show(){
        printf("[decoder] summary printout with %d components\n",(int) components.size());
        for(int loop = 0; loop < components.size(); loop++){
            components[loop]->summary();
        }
    }

    void  decoder::reset(){
        for(int loop = 0; loop < components.size(); loop++){
            components[loop]->reset();
        }
    }
   
   void  decoder::write(hipo::event &event){
      event.reset();
      for(int loop = 0; loop < components.size(); loop++){
          std::vector<hipo::composite>  banks = components[loop]->getBanks();
          for(int b = 0; b < banks.size(); b++){
            if(banks[b].getRows()>0) event.addStructure(banks[b]);
          }
      }
      if(doStatistics==true){
         processStatistics(event.getSize());
      }
      reset();
   }

   void   decoder::processStatistics(int eventSize){
        eventSizeStats.push_back(eventSize);
        //printf("processing: %d, %d %d\n",eventSizeStats.size(),eventStats.size(),statisticsCounter);
        if((eventSizeStats.size() + 1)%agregationCount==0){
            double vec = 0; for(int i= 0; i < eventSizeStats.size(); i++) vec += eventSizeStats[i];
            eventStats.push_back(vec/eventSizeStats.size()); eventSizeStats.clear();
        }
        statisticsCounter++;
        if(statisticsCounter%statsPrintFrequency==0){
          bench.pause();
          showStats();
          bench.reset();
          bench.resume();
          eventStats.clear();
        }
   }

}
