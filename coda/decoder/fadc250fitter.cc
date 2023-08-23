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
 * File:   table.cc
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */


#include "table.h"
#include <fstream>
#include "fadc250fitter.h"
#include "chart/ascii.h"
//------------------------------------------------------------------------------

namespace coda {

  int  fadc250::maximum(){
    int max = 0; 
    int size = getLength();
    for(int i = 0; i < size; i++) max = std::max(max,getBin(i));
    return max;
  }

  void fadc250::fit(fadc_t &coef){
    t0 = 0;  adc = 0;  ped = 0; int baseline = 0; int rms = 0;
    int thresholdCrossing = 0;  pulsePeakValue = 0;  pulsePeakPosition = 0;
    int pulseWidth = 0; int tcourse = 0; int tfine = 0;
    double noise = 0.0; int pedistalMaxBin = 5;int pedistalMinBin = 1;
    int    tstart = pedistalMaxBin+1;

    int    tcross = 0;
    int    pmax   = 0;
    int    ppos   = 0;
     
    if(coef.ped!=0) ped = coef.ped;

      if(getLength()<p2+1 && coef.ped==0) {
          for (int bin = 0; bin < getLength(); bin++) {
            pedsum += getBin(bin);
          }
          pedestal=pedsum/getLength();
          return;
      }
      /*If the mode is full pulse mode - proceed here*/
      if (coef.ped==0) {
            tstart = p2+1;
            for (int bin = p1+1; bin < p2+1; bin++) {
                pedsum += getBin(bin);
                noise  += getBin(bin) * getBin(bin);
            }
            baseline = ((double) pedsum)/ (p2 - p1);
            ped = pedsum = pedsum/(p2-p1);	//(int) baseline;
            rms = std::sqrt(noise / (p2 - p1) - baseline * baseline);
        }
        // find threshold crossing
        for (int bin=tstart; bin<getLength(); bin++) {
            if(getBin(bin)>ped+coef.tet) {
                tcross=bin;
                thresholdCrossing=tcross;
                break;
            }
        }
        
        if(tcross>0) {
           for (int bin=std::max(0,tcross-coef.nsb); bin<std::min(getLength(),tcross+coef.nsa+1); bin++) { // sum should be up to tcross+nsa (without +1), this was added to match the old fit method
                adc += getBin(bin)-ped;
                if(bin>=tcross && getBin(bin)>pmax) {
                    pmax=getBin(bin);
                    ppos=bin;
                }
            }
            pulsePeakPosition=ppos;
            pulsePeakValue=pmax;

            double halfMax = (pmax+baseline)/2;
            int s0 = -1;
            int s1 = -1;
            for (int bin=tcross-1; bin<std::min(getLength()-1,ppos+1); bin++) {
                if (getBin(bin)<=halfMax && getBin(bin+1)>halfMax) {
                    s0 = bin;
                    break;
                }
            }
            for (int bin=ppos; bin<std::min(getLength()-1,tcross+coef.nsa); bin++) {
                if (pulse[bin]>halfMax && pulse[bin+1]<=halfMax) {
                    s1 = bin;
                    break;
                }
            }
            if(s0>-1) {
                int a0 = getBin(s0);
                int a1 = getBin(s0+1);
                // set course time to be the sample before the 50% crossing
                tcourse = s0;
                // set the fine time from interpolation between the two samples before and after the 50% crossing (6 bits resolution)
                tfine   = ((int) ((halfMax - a0)/(a1-a0) * 64));
                t0      = (tcourse << 6) + tfine;
            }
            if(s1>-1 && s0>-1) {
                int a0 = getBin(s1);
                int a1 = getBin(s1+1);
                pulseWidth  = s1 - s0;
            }
            adc_corrected = adc + ped*(coef.nsa+coef.nsb);
        }

   }

   void fadc250::show(){
      printf("fadc250: n = %4d, adc = %5d, time = %8d\n",getLength(),adc_corrected,t0);
   }

  
  void fadc250::graph(){
    std::vector<double> series;
    for(int p = 0; p < getLength(); p++)  series.push_back(getBin(p));
    //}
    //ascii_chart::plot(pulse);
    //ascii::Asciichart asciichart(std::vector<std::vector<double>>{series});
    ascii::Asciichart asciichart({{"FADC",series}});
    std::cout << asciichart.show_legend(true).height(10).Plot();
  }
  
  void fadc250::csv(coda::fadc_t &coef){
    printf("pulse: %d,%e,%d,%d,%d,", getLength(),coef.ped,coef.nsa,coef.nsb, coef.tet);
    printf("%d,%d,%d,%d",getLength(),adc_corrected,t0,ped);
    for(int k = 0; k < getLength(); k++) printf(",%d",getBin(k));    
    printf("\n");
  }
  
  int  fadc250::getMax(){
    int max = 0;
    for(int k = 0; k < getLength(); k++) if(getBin(k)>max) max=getBin(k);
    return max;
  }
  
  void fadc250::csv(){
    int max = getMax();
    max = max*1.001;
    std::vector<double> vec;
    for(int k = 0; k < getLength(); k++) vec.push_back(((double) getBin(k))/max);
    double summ = 0;
    for(int i =0; i < 5; i++) summ += vec[i+1];
    double ped = summ/5;

    if(ped<0.2&&getLength()==48) {
      printf("pulse_selfnorm: %d", getLength());
      for(int k = 0; k < getLength(); k++) printf(",%.4f",vec[k]);
      printf("\n");
    }
    
    //if(max<1700){
      /*
      printf("pulse_selfnorm: %d", getLength());
      //printf("%d,%d,%d,%d",getLength(),adc_corrected,t0,ped);
      for(int k = 0; k < getLength(); k++) printf(",%.4f",((double) getBin(k))/max);
      printf("\n");
      printf("pulse_totalnorm: %d", getLength());
      //printf("%d,%d,%d,%d",getLength(),adc_corrected,t0,ped);
      for(int k = 0; k < getLength(); k++) printf(",%.4f",((double) getBin(k))/1700);
      printf("\n");
    }*/
  }
  
  void fadc250::print(){
    printf("%d,%d,%d",getLength(),adc_corrected,t0);
    for(int k = 0; k < getLength(); k++) printf(",%d",getBin(k));    
    printf("\n");
  }

  void   fadc250::print(coda::fadc_t &coef){
     printf("entry: ped = %8.4f, nsa = %5d, nsb = %5d, tet = %5d\n", coef.ped,coef.nsa,coef.nsb, coef.tet);
  }
}
