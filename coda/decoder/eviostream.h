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
/******************************************************************************
 * * File:   eviostream.h
 * Author: gavalian
 *
 * Created on July 19, 2023, 10:14 AM
 */
#ifndef __EVIOSTREAM__
#define __EVIOSTREAM__

#include "reader.h"
#include "writer.h"
#include "utils.h"
#include "evio.h"
#include "container.h"

namespace evio {

    /**
     * @brief stream class for getting evio events in bulk from evio file in
     * multi-threaded environment. The pull method has a mutex lock so all calls
     * are thread safe.
     */
    class eviostream {
        private:
           
           hipo::writer  writer;
           int           evioHandle;
           int           MAX_BUFFER = 1024*100;
           std::mutex obj;
        public:
            eviostream(){ }
            virtual ~eviostream(){};
            void open(const char *file){
                writer.open("output.h5");
                int err = evOpen((char*) file,"r",&evioHandle);
                printf("evOpen: error code = %d\n",err);
            }

            bool pull(std::vector<container> &events){
                std::unique_lock<std::mutex> lock(obj);
                bool success = true;
                for(int i = 0; i < events.size(); i++){
                    int err = evRead(evioHandle,&events[i].buffer[0],MAX_BUFFER);
                    if(err!=S_SUCCESS) { events[i].buffer[0] = 0; success = false;}
                    //printf(">>>> reading evne # %d, err code = %d size = %d\n",i,err, events[i].buffer[0]);
                }
                return success;
            }
            
            void push(hipo::event &evt){
                writer.addEvent(evt);
            }
            void close(){
                writer.close();
            }
    };
}

#endif
