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

#ifndef HIPO_COMPONENT_H
#define HIPO_COMPONENT_H

#include <map>
#include <unordered_map>
#include <tuple>
#include "table.h"
#include "fadc250fitter.h"
#include "bank.h"


namespace coda {

struct eviodata_t {
   int crate; // this is fragment number
   int tag; // this is the tag number of composite bank
   int offset; // offset to the start of the data
   int length;
   const char *buffer;// pointer to the evio buffer
};

class component {

   protected:

      coda::table   table;
      coda::fitter  fitter;
      std::string   name = "unknown";

      std::map<int,int> crateMap;

      std::vector<hipo::composite> banks;
   
      virtual void rehash();
      virtual void setBankSize(int n);
      void  decode_fadc250(eviodata_t &data, coda::fitter &__fitter, coda::table &__table, hipo::composite &bank);
      void  decode_tdc(    eviodata_t &data, coda::table &__table, hipo::composite &bank);
      void  initBanks(std::initializer_list<std::tuple<int,int,const char *,int> > desc);

   public:
      
      component(){};
      component(const char *fileTable);
      component(const char *filetable, const char *filefitter);
      
      virtual void addBank(int group, int item, const char *format, int size);

      void  setName(const char *__name){ name = __name;}
      
      virtual       ~component(){};
      virtual bool   hasCrate(int crate);
      virtual bool   readTable(const char *file);
      virtual bool   readFitter(const char *file);
      virtual void   reset();
      virtual void   summary();
      virtual void   decode(eviodata_t &evio);
      virtual void   init();
      virtual std::set<int> &keys();
      
      std::vector<hipo::composite>  &getBanks(){return banks;}
      std::string                   &getName(){ return name;}
};
} // end of namespace

#endif /*HIPO_TABLE_H*/

