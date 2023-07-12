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
 * File:   component.h
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */

#ifndef HIPO_FADC250FITTER_H
#define HIPO_FADC250FITTER_H

#include <map>
#include <vector>
#include <unordered_map>
#include "table.h"
//#include "ascii_chart.hpp"

namespace coda {

class fadc250 {

private:
  int fadcamp = 0;
  int pedestal = 0;
  int pedsum;
  int p1,p2;
  int t0,adc, ped;
  int pulsePeakValue ,  pulsePeakPosition ;
  int adc_corrected;
  // functions
  
public:
  
  std::vector<uint8_t>  pulse;
  fadc250(){ pulse.resize(320); p1 = 1; p2 = 15; pedsum = 0;}
  virtual ~fadc250(){};
  
  void   fit(fadc_t &coef);
  int    getAdc(){ return adc_corrected;}
  float  getTime(){return t0;}
  int    getPedestal(){ return pedestal;}
  
  int    maximum();
  int    getLength(){ return *(reinterpret_cast<uint16_t *>(&pulse[0]));}
  int    getMax();
  int    getBin(int bin){return *(reinterpret_cast<uint16_t *>(&pulse[2+bin*2]));}
  void   setLength(int length){*(reinterpret_cast<uint16_t *>(&pulse[0])) = (uint16_t) length;}
  void   setBin(int bin, uint16_t value){ *(reinterpret_cast<uint16_t *>(&pulse[2+bin*2]))=value;};
  void   show();
  void   graph();
  void   print();
  void   csv(coda::fadc_t &coef);
  void   csv();
  void   print(coda::fadc_t &coef);
};

} // end of namespace

#endif /*HIPO_TABLE_H*/

