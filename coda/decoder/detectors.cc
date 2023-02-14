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
#include "detectors.h"


namespace coda {
  /**
   * @brief Construct a new component ec::component ec object
   * 
   * @param filetable - table name for translation (txt file)
   * @param filefitter - table name for fadc pulse fit parameters (txt file)
   */
  component_ec::component_ec(const char* filetable, const char *filefitter) : component(filetable, filefitter) {
    //addBank(11,3,"bbsbif",1024); // this is an adc bank 
    setName("ec"); setBankSize(2);
  }

  void   component_ec::decode(eviodata_t &evio){
    //printf("[%s] decoder is called for crate = %4d with tag = %d\n",name.c_str(),evio.crate, evio.tag);
    if(evio.tag==0xe101) {
      decode_fadc250(evio,fitter,table,banks[1]);
      //printf(" decoded bank %X, now - rows = %d\n",evio.tag,banks[1].getRows());
    }
  }

void component_ec::init(){
  initBanks( {{ 11,1,"bbsbil",512},{11,2,"bbsbifs",512}} );
}

  /**
  * @brief Construct a new component dc::component dc object
  * 
  * @param filetable - table name for translation (txt file)
  */
  component_dc::component_dc(const char* filetable) : component(filetable) {    
    setName("dc"); setBankSize(1);
  }

  void   component_dc::decode(eviodata_t &evio){
    if(evio.tag==0xe116) {
      decode_tdc(evio,table,banks[0]);
    }
  }

  void component_dc::init(){
    initBanks( {{ 12,1,"bbsbil",512}} );
  }
}
