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
 * File:   component.cc
 * Author: gavalian
 *
 * Created on April 12, 2017, 10:14 AM
 */


#include "table.h"
#include <fstream>
#include "component.h"


namespace coda {

  component::component(const char *fileTable){ 
    readTable(fileTable); 
    rehash();
  }

  component::component(const char *filetable, const char *filefitter){ 
    readTable(filetable); 
    readFitter(filefitter);
    rehash();
  }

void component::setBankSize(int n){
    banks.clear();
    for(int i = 0; i < n; i++){
        hipo::composite bank;
        banks.push_back(bank);
    }
}

void  component::initBanks(std::initializer_list<std::tuple<int,int,const char *,int> > desc){
    if(desc.size()!=banks.size()){ printf("component:: missmatch between banks size and descriptors [%d,%d]\n",(int) banks.size(), (int) desc.size()); return; }
    int counter = 0;
    std::initializer_list< std::tuple<int,int,const char *,int>>::iterator it;  // same as: const int* it
    for ( it=desc.begin(); it!=desc.end(); ++it){
        banks[counter].parse(std::get<0>(*it),std::get<1>(*it),std::get<2>(*it),std::get<3>(*it));
        counter++;
    }
}

void component::init(){

}
void component::rehash(){
     
}

void   component::addBank(int group, int item, const char *format, int size){      
       hipo::composite bank;
       int capacity = banks.size();
       banks.push_back(bank);
       banks[capacity].parse(group,item,std::string(format),size);
       //bank.parse(group,item,std::string(format),size);
       //banks.push_back(std::move(bank));
}

void   component::decode(eviodata_t &evio){
    //if(evio.tag==0xe101) decode_fadc250(evio,fitter,table,banks[1]);
    printf(" this method is not implemented for the supper class\n");
}

bool   component::hasCrate(int crate){ return true;}
bool   component::readTable(const char *file){ table.read(file); return true;}
bool   component::readFitter(const char *file){fitter.read(file); return true;}
void   component::reset(){ for(int k = 0; k < banks.size(); k++) banks[k].reset();}

std::set<int> &component::keys(){
    return table.getKeys();
}

void   component::summary(){
    printf("[component] [%9s] banks (%3d) > sizes :", name.c_str(),(int) banks.size());
    for(int k = 0; k < banks.size(); k++) printf("%2d -> %5d,",k,banks[k].getRows());
     printf("\n");   
    for(int k = 0; k < banks.size(); k++) banks[k].show();

}
//########################################################################################################
// This is part where ugly code resides for standard ADC & TDC decoding from EVIO composite banks
// the decode FADC parses banks with TAG=0xe101 (57601) composite banks, the format is ()
//########################################################################################################
void  component::decode_fadc250(eviodata_t &data, coda::fitter &__fitter, coda::table &__table, hipo::composite &bank){
    int pos = data.offset;
    bool doLoop = true;
    fadc250  fadc;
    int   row = bank.getRows();

    while(doLoop==true){
            uint8_t        slot = *reinterpret_cast<const uint8_t*>( &data.buffer[pos]);
            //uint32_t    tnumber = *reinterpret_cast<const uint32_t*>(&buffer[pos + 1]);
            uint64_t  timestamp = *reinterpret_cast<const uint64_t*>(&data.buffer[pos + 5]);
            uint32_t    nrepeat = *reinterpret_cast<const uint32_t*>(&data.buffer[pos + 13]);
            pos += 17;
            descriptor_t desc;
            if(nrepeat>1000) break;
            for(int n =  0; n < nrepeat; n++){
                uint8_t   channel =  *reinterpret_cast<const uint8_t*>( &data.buffer[pos]);
                uint32_t  nsamples = *reinterpret_cast<const uint32_t*>(&data.buffer[pos + 1]);
                pos+= 5;
                bool contains = __fitter.contains(data.crate,slot,channel);
                if(contains==true){
                    memcpy(&fadc.pulse[0],&data.buffer[pos],nsamples*2);
                    fadc.setLength(nsamples);
                    fadc_t params = __fitter.get(data.crate,slot,channel);
                    //printf("********************************\n");
                    
                    //fadc.fit(params);
                    //fadc.print(params);
                    //fadc.show();
                    
                    params.ped = 0.0;
                    //printf(">>>> \n");
                    fadc.fit(params);
                    //fadc.print(params);
                    //fadc.show();
                    //if(fadc.getTime()<0.001) fadc.csv(params);
		   // fadc.csv();
                    //fadc.print(__fitter.get(data.crate,slot,channel));
                    //fadc.print(params);
                    //fadc.print();
                    //if(fadc.getTime()<0.001) fadc.csv(__fitter.get(data.crate,slot,channel));

                    desc.crate = data.crate;
                    desc.slot = slot;
                    desc.channel = channel;
                    if(table.contains(desc)==true){
                        table.decode(desc);
                        bank.putInt(row,0,desc.sector);
                        bank.putInt(row,1,desc.layer);
                        bank.putInt(row,2,desc.component);
                        bank.putInt(row,3,desc.order);
                        bank.putInt(row,4,fadc.getAdc());
                        bank.putFloat(row,5,fadc.getTime());
                        bank.putInt(row,6,fadc.getPedestal());
                        row++;
                    }
                    //bank.setRows(row);                    
                    //printf(" iter : %4d (%4d, %4d, %4d), pulse samples = %d  max = %5d -> %s\n",n,
                    //crate, slot, channel,nsamples, fadc.maximum() , contains?"true":"false");
                    
                } else {
                   // printf(">>>>  error decoding FADC 250 [%d,%d,%d]\n",data.crate,slot, channel);
                }
                pos += 2*nsamples;
            }
            if((pos+17 - data.offset ) < data.length) doLoop = false;
         }
         //bank.show();
         //bank.print();
}
//********************************************************************************************
//- Parsing of TDC bank written in composite EVIO format, TAG=0xe116 (57622)
//********************************************************************************************
void  component::decode_tdc(eviodata_t &data, coda::table &__table, hipo::composite &bank){
       int   pos = data.offset;
       bool  doLoop = true;
       int   row = bank.getRows();
       descriptor_t desc;
       desc.crate = data.crate;
       //printf(">>>>>>>> decoding tdc : %d\n", data.length);
       while(doLoop==true){
          uint8_t        slot = *reinterpret_cast<const uint8_t*>( &data.buffer[pos]);
          //uint32_t    tnumber = *reinterpret_cast<const uint32_t*>(&buffer[pos + 1]);
          uint64_t  timestamp = *reinterpret_cast<const uint64_t*>(&data.buffer[pos + 5]);
          uint32_t    nrepeat = *reinterpret_cast<const uint32_t*>(&data.buffer[pos + 13]);
          pos += 17;
          //printf("--- n-repeat : %d\n", nrepeat);
         if(nrepeat>1000) break;
          
          //printf  ("\n >>>> slot = %d, N = %d\n",slot, nrepeat);
          for(int n =  0; n < nrepeat; n++){
             uint8_t  channel =  *reinterpret_cast<const uint8_t*>(  &data.buffer[pos]);
             uint16_t     tdc =  *reinterpret_cast<const uint16_t*>( &data.buffer[pos+1]);
             pos += 3;
             desc.slot = slot;
             desc.channel = channel;
             //printf("\t\t channel = %d tdc = %d\n",channel,tdc);
             
             if(table.contains(desc)==true){
             //if(1){   
                table.decode(desc);
                bank.putInt(row,0,desc.sector);
                bank.putInt(row,1,desc.layer);
                bank.putInt(row,2,desc.component);
                bank.putInt(row,3,desc.order);
                bank.putInt(row,4,tdc);
                bank.putLong(row,5,timestamp);
                row++;
             } else {
                printf(" error in decoder TDC : "); table.print(desc);
             }
           } 
          if((pos+17 - data.offset ) > data.length) doLoop = false;
          //if((pos+17 - offset ) < length) doLoop = false;
       }
        //printf(">>>>>>>> decoding done tdc : %d , offset %d pos %d, moved %d\n", 
        //data.length,data.offset,pos,pos-data.offset);
    }
    //************************************************************************************************
    // decoding TDC bank TAG= 0xe107 (57607), the TDCs are stored as integer array (32 bit) 
    // with structure:
    //  slot - bits 27-31 mask = 
    //  channel - bits 19-25
    //  value - bits 0-18
    //************************************************************************************************
    void  component::decode_tdc_57607( eviodata_t &data, coda::table &__table, hipo::composite &bank){
        int   pos = data.offset;
        bool  doLoop = true;
        int   row = bank.getRows();
        descriptor_t desc;
        desc.crate = data.crate;
        while(doLoop==true){
            uint32_t word = *reinterpret_cast<const uint32_t*>( &data.buffer[pos]);
            int    slot = (word>>27)&0x0000001F;
            int channel = (word>>19)&0x0000007F;
            int   value = (word)&0x0007FFFF;
            desc.slot = slot;
            desc.channel = channel;
            if(table.contains(desc)==true){
                table.decode(desc);
                bank.putInt(row,0,desc.sector);
                bank.putInt(row,1,desc.layer);
                bank.putInt(row,2,desc.component);
                bank.putInt(row,3,desc.order);
                bank.putInt(row,4,value);
                row++;
            } else {
                //printf(" error in decoder : ");
                 //table.print(desc);
                 //printf("position = %d, offset %d, length = %d, dist = %d, value = %X\n",pos,
                 //data.offset, data.length, pos +4 - data.offset, word);
            }
            pos += 4;
            if((pos+4 - data.offset ) > data.length) doLoop = false;
        }
    }
}
