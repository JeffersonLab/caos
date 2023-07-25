#ifndef __EVIO_CONTAINER__
#define __EVIO_CONTAINER__

namespace coda {

struct eviodata_t {
        int crate; // this is fragment number
        int tag; // this is the tag number of composite bank
        int offset; // offset to the start of the data
        int length;
        const char *buffer;// pointer to the evio buffer
};

}
namespace evio {

    

    class container {
        private:
           int position = -1;
           int nextposition = 0;
           int length   =  0;
           int banklength = 0;
           int crate = 0;
        public:
            std::vector<uint32_t> buffer;
            container(){};
            container(int size){ buffer.resize(size);};
            virtual ~container(){};

            void init(){
                position = 2;
                nextposition = 2;
                length   = buffer[0]+1;
            }
            
            void copy(uint32_t *eviobuffer){
                int size = eviobuffer[0];
                if(buffer.size()<size) {printf("resising to size = %d\n",size*4); buffer.resize(size+1024);}
                memcpy(reinterpret_cast<void *>(&buffer[0]),reinterpret_cast<void *>(eviobuffer),size*4);
            }

            std::vector<uint32_t> &getBuffer( ){return buffer;}
            int size(){ return buffer[0];}
            /*
             * this portion advvances in the EVIO buffer by stoping at the BANK instances
             * to get a composite bank inside of the bank, use link method.
             * to start from the top of the event call method init() first.
             */
            bool next(){
                position = nextposition;
                if(position>=buffer[0]) return false;
                int  nwf = buffer[position] + 1; /* the number of words in fragment */
                int tagf = (buffer[position+1]>>16)&0xffff;
                int padf = (buffer[position+1]>>14)&0x3;
                int typf = (buffer[position+1]>>8)&0x3f;
                int numf =  buffer[position+1]&0xff;
                
                while(typf!=0xe && typf!=0x10){
                    position += nwf;
                    if(position>length) return false;
                    nwf  = buffer[position] + 1; /* the number of words in fragment */
                    tagf = (buffer[position+1]>>16)&0xffff;
                    padf = (buffer[position+1]>>14)&0x3;
                    typf = (buffer[position+1]>>8)&0x3f;
                    numf =  buffer[position+1]&0xff;
                }
                nextposition = position + nwf;
                banklength = nwf;
                crate = tagf;
                //printf("position %8d : tag = %5d, num = %5d, type = %5X\n",position,tagf,numf,typf);
                return true;
            }
            /*
             * this portion of the code finds composite bank inside of the BANK represetned
             * by position variable in the class.
             */
            bool link(coda::eviodata_t &eptr){

                int  ind = position + 2;
                int lenb = 2;
                //printf("looking into %d bank size = %d (tag = %d)\n",position, banklength,(buffer[position+1]>>16)&0xffff);
                while(lenb<banklength){
                    int nwb  = buffer[ind] + 1;
                    int tagb = (buffer[ind+1]>>16)&0xffff;
                    int padb = (buffer[ind+1]>>14)&0x3;
                    int typb = (buffer[ind+1]>>8)&0x3f;
                    int numb =  buffer[ind+1]&0xff;
                    //printf("inside while (%d, %d) type = %d\n",tagb,numb,typb);
                    if(typb != 0xf){
                        int nbytes = (nwb-2)<<2;
                        int ind_data = ind+2;
                        ind  += nwb;
                        lenb += nwb;
                    } else {
                        //printf(" found composite bank tag = %5X (%5d) , num = %5d \n",tagb,tagb,numb);
                        int ind2 = ind+2; /* index of the tagsegment (contains format description) */
                        int len2 = (buffer[ind2]&0xffff) + 1; /* tagsegment length */
                        int ind3 = ind2 + len2; /* index of the internal bank */
                        int pad3 = (buffer[ind3+1]>>14)&0x3; /* padding from internal bank */
                        int nbytes = ((nwb-(2+len2+2))<<2)-pad3; /* bank_length - bank_header_length(2) - tagsegment_length(len2) - internal_bank_header_length(2) */
                        int ind_data = ind+2+len2+2;
                        
                        eptr.crate = crate;
                        eptr.tag   = tagb;
                        eptr.offset = ind_data*4;
                        eptr.length = nbytes;
                        eptr.buffer = reinterpret_cast<const char *>(&buffer[0]);
                        return true;
                    }
                }
                eptr.crate  = -1;
                eptr.tag    = 0;
                eptr.offset = -1;
                eptr.length = 0;
                eptr.buffer = reinterpret_cast<const char *>(&buffer[0]);
                return false;
            }

            static std::vector<container> create(int size, int length){
                std::vector<container> vec;
                for(int i = 0; i < size; i++) {
                    container c(length);
                    vec.push_back(c);
                }
                return vec;
            }
    };
} // __ end of namespace evio

#endif