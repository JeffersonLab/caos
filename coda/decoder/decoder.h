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
 * File:   decoder.h
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */

#ifndef HIPO_DECODER_H
#define HIPO_DECODER_H

#include <map>
#include <vector>
#include "table.h"
#include "bank.h"
#include "detectors.h"
#include "event.h"
#include "chart/ascii.h"
#include "utils.h"
#include "container.h"


namespace coda {

class decoder {
    
    protected:

       std::vector<coda::component *> components;
       std::map<int,int>              crateMap;
       std::vector<std::set<int>>     componentSets;

       
       std::vector<double>            eventSizeStats;
       std::vector<double>            eventStats;
       
       bool   doStatistics;

       int      statisticsHeight  = 24;
       int      statisticsCounter = 0;
       int        agregationCount = 400;
       int    statsPrintFrequency = 1000;

        hipo::benchmark   bench;

       static void print(int x);
       
       void   constructDetectorSet();
       void   showStats();
       void   plotStats();
       void   processStatistics(int eventSize);
       hipo::node    evionode;//({1,11,1,1024*1024});
       evio::container container;

    public:

    decoder(){};
    virtual ~decoder(){ doStatistics = false; bench.setName("decoder"); bench.resume();};

    void  stats(bool flag){ doStatistics = flag;}

    void  initialize();
    void  decode(coda::eviodata_t &evio);
    void  decode(uint32_t *evioBuffer);
    void  write(hipo::event &event);
    void  write(hipo::event &event, evio::container &evio);
    void  reset();
    void  show();
    void  showKeys();

};

} // end of namespace

#endif /*HIPO_TABLE_H*/

