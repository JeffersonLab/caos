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
 * File:   table.h
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */

#ifndef HIPO_TABLE_H
#define HIPO_TABLE_H

#include <map>
#include <set>
#include <unordered_map>
#include "unordered_dense.h"

namespace coda {
 
typedef struct descriptor_t {
   
   int crate;
   int slot;
   int channel;
   int sector;
   int layer;
   int component;
   int order;
   descriptor_t() : crate(0),slot(0),channel(0),sector(0),layer(0),component(0),order(0) {}
   descriptor_t(int cr, int sl, int ch) 
          : sector(0),layer(0),component(0),order(0) {crate = cr; slot = sl; channel = ch;}
}  descriptor_t;

typedef struct tdc_t {
   descriptor_t desc;
   int          tdc;
   tdc_t() : desc(0,0,0), tdc(0) {}
} tdc_t;


typedef struct fadc_t {
   double  ped;
   int     nsa;
   int     nsb;
   int     tet;
   fadc_t(): ped(0.0), nsa(0), nsb(0), tet(0) {}
   fadc_t(double __ped, int __nsa, int __nsb, int __tet){ ped = __ped; nsa = __nsa; nsb = __nsb; tet = __tet;}
} fadc_t;

class fitter {
   private:
      std::map<long,fadc_t>  fadcmap;
   public:
      fitter(){};
      virtual  ~fitter(){};
      long     encode (int crate, int slot, int channel);
      bool     contains(int crate, int slot, int channel);
      fadc_t  &get(int crate, int slot, int channel);
      void     read(const char *filename);
};

class table {
    
    private:
     //std::map<long,long>  translation;
     std::unordered_map<long,long>  translation;
     //ankerl::unordered_dense::map<long,long> translation;
     std::set<int>        crateKeys;
     //std::unordered_map<long,long>  translation;
    public:

     table(){};
     virtual ~table(){};
     
     long   encode   ( int id1, int id2, int id3, int id4);
     void   decode   ( descriptor_t &desc);
     bool   contains ( descriptor_t &desc);
     void   insert   ( descriptor_t &desc);
     void   read     ( const char *filename);
     void   print    ( descriptor_t &desc);

     std::set<int>   &getKeys();
};

} // end of namespace

#endif /*HIPO_TABLE_H*/
