/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;

import java.util.ArrayList;
import java.util.List;

import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Converter_MultiClass {
    
    public static void analyzer(String file){
        
        CompositeNode nodeRC = new CompositeNode(  5, 1,  "b", 10);
        CompositeNode nodeET = new CompositeNode(  5, 2,  "b", 10);
        HipoReader r = new HipoReader(file);
        Event e = new Event();
        
        H1F HT = new H1F("ht",6,0.5,6.5);
        H1F HE = new H1F("he",6,0.5,6.5);
        HT.attr().setLineColor(4);
        TGCanvas c = new TGCanvas();
        c.draw(HE).draw(HT,"same");
        while(r.hasNext()){
            r.next(e);
            e.read(nodeET, 5, 2);
            e.read(nodeRC, 5, 1);
            for(int i = 1; i <= 6; i++){
                if(nodeRC.getInt(0, i)>0) HT.fill(i);
                if(nodeET.getInt(0, i)>0) HE.fill(i);
            }
        }
    }
    
    public static int[] convertTriggerLong(long  bits){
        int[] trigger = new int[7];
        for(int i = 0; i < trigger.length; i++) {
            trigger[i] = 0;
            if( ((bits>>i)&(1L)) != 0L) trigger[i] = 1;
            //System.out.println(Arrays.toString(trigger));
        }
        return trigger;
    }
    public static boolean contains(int[] array, int value){
        for(int i = 0; i < array.length; i++) if(array[i]==value) return true;
        return false;
    }
    
    public static void convertEC(Bank bank, CompositeNode node, int... sectors){
        node.setRows(0);
        int nrowsec = bank.getRows();
        if(bank.getRows()<4000){
            int counter = 0;
            
            for(int row = 0; row < nrowsec; row++){
                int sector = bank.getInt("sector", row);
                if(Level3Converter_MultiClass.contains(sectors, sector)==true){
                    node.setRows(counter+1); 
                    node.putByte(  0, row, (byte) bank.getInt("sector", row));
                    node.putByte(  1, row, (byte) bank.getInt("layer", row));
                    node.putShort( 2, row, (short) bank.getInt("component", row));
                    node.putByte(  3, row, (byte) bank.getInt("order", row));
                    node.putInt(   4, row,  bank.getInt("ADC", row));
                    counter++;
                }
            }
        }
    }

    public static int getTrackIndex(Bank TrackBank, int pIndex){
        int trIndex=-1;
        for (int row=0; row< TrackBank.getRows();row++){
            int trackPIndex=TrackBank.getInt("pindex", row);
            if (trackPIndex==pIndex){trIndex=row;}
        }
        return trIndex;
    }

    public static float getCALE(Bank CALBank, int pIndex){
        float energy=0;
        for (int row=0; row<CALBank.getRows();row++){
            int trackPIndex=CALBank.getInt("pindex", row);
            if (trackPIndex==pIndex){energy+=CALBank.getFloat("energy", row);}
        }
        return energy;
    }

    public static List<Integer> getPIndices_fSector(Bank TrackBank, int sector){
        List<Integer> pIndices = new ArrayList<Integer>();
        for (int row=0; row< TrackBank.getRows();row++){
            int trackPIndex=TrackBank.getInt("pindex", row);
            int trSector=TrackBank.getInt("sector", row);
            if (trSector==sector){pIndices.add(trackPIndex);}
        }
        return pIndices;
    }
    
    public static void convertDC(Bank bank, CompositeNode node, int... sectors){
        node.setRows(0);
        int nrowsdc = bank.getRows();
        if(bank.getRows()<4000){

            //node.setRows(nrowsdc);
            int counter = 0;
            for(int row = 0; row < nrowsdc; row++){
                int sector = bank.getInt("sector", row);
                if(Level3Converter_MultiClass.contains(sectors, sector)==true){
                    node.setRows(counter+1);                
                    node.putByte(  0, counter, (byte) bank.getInt("sector", row));
                    node.putByte(  1, counter, (byte) bank.getInt("layer", row));
                    node.putShort( 2, counter, (short) bank.getInt("component", row));
                    node.putByte(  3, counter, (byte) bank.getInt("order", row));
                    node.putInt(   4, counter,  bank.getInt("TDC", row));
                    counter++;
                }
            }
        }
    }
    public static int getTriggerSector(int[] trigger){
        if(trigger[0]<1) return 0;
        for(int s = 1; s < trigger.length; s++) 
            if(trigger[s]>0) return s;                
        return 0;
    }
    
    public static void convertFile(String file, String output){
        HipoReader r = new HipoReader(file);
        HipoWriter w = HipoWriter.create(output, r);
        Event e = new Event();
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        
        while(r.hasNext()){
            
            r.nextEvent(e);
            e.read(banks);
            
            e.read(dsts);

            //V3

            //loop over sectors
            for (int sect = 1; sect < 7; sect++) {
                // get a list of pIndices in the sector
                List<Integer> pIndices = Level3Converter_MultiClass.getPIndices_fSector(dsts[1], sect);

                //pid=-1 denotes no track in sector
                int pid=-1;
                float nEdep=0;
                //check if there's at least 1 track in sector
                if (pIndices.size() > 0) {
                    pid = dsts[0].getInt(0, 0);
                    
                    //loop over particle indices in sector to check if there's an electron in sector
                    for (int i = 0; i < pIndices.size(); i++) {
                        if (dsts[0].getInt(0, i) == 11) {
                            pid = 11;
                        }
                        //check if we have neutral pids
                        if (dsts[0].getInt(0, i) == 22 || dsts[0].getInt(0, i) == 2212 || dsts[0].getInt(0, i) == 0) {
                            nEdep+=Level3Converter_MultiClass.getCALE(dsts[2],i);
                        }
                    }
                }

                //for each data entry we record trigger
                //could help with some logic in training data
                //eg requiring conventional trigger to be wrong
                long bits = banks[2].getLong("trigger", 0);
                int[] trigger = Level3Converter_MultiClass.convertTriggerLong(bits);
                int trigSector = Level3Converter_MultiClass.getTriggerSector(trigger);

                int[] labels = new int[] { pid, trigSector, sect };

                // get DC and EC from the sector
                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);

                Boolean hasNeutral=false;
                //hasNeutral means there's one or more photon 
                //with >1 GeV E dep in calorimeters in total
                if(nEdep>1){hasNeutral=true;}

                //tag 0 is trash
                int tag=0;

                //tag=1 means there's an electron
                if(pid==11){tag=1;}

                //tag 2 means there's a negative pion but no neutral
                //biggest source of bg for electrons
                if(pid==-211 && !hasNeutral){tag=2;}
                
                //tag 3 means there's another particle (not el or pi-) but no neutral
                if(pid!=11 && pid!=-211 && !hasNeutral){tag=3;}

                 //tag 4 means there's a pion and neutral
                if(pid==-211 && hasNeutral){tag=4;}

                //tag 5 means there's another particle (not el not pion) and neutral
                if(pid!=11 && pid!=-211 && hasNeutral){tag=4;}
                
                //tag 6 means there's no track and a neutral
                if(pid==-1 && hasNeutral){tag=6;}

                // nodeDC.print();
                // nodeEC.print();

                Node tnode = new Node(5, 4, labels);

                if (nodeEC.getRows() > 0 && nodeDC.getRows() > 0 && tag>0) {
                    e.write(nodeEC);
                    e.write(nodeDC);
                    e.write(tnode);

                    e.setEventTag(tag);
                    w.addEvent(e);
                }


            }
        }
        
        w.close();
    }
    
    public static void extract(String file){
        HipoReader r = new HipoReader(file);
        Event e = new Event();
        
        CompositeNode nodeRC = new CompositeNode(  5, 1,  "b", 10);
        CompositeNode nodeET = new CompositeNode(  5, 2,  "b", 10);
        
        while(r.hasNext()){
            r.next(e);
            e.read(nodeET, 5, 2);
            e.read(nodeRC, 5, 1);
            
            System.out.printf( "--- %d %d\n", nodeRC.getInt(0, 0), nodeET.getInt(0, 0));
        }
    }
    public static void main(String[] args){        
        

        /*String file = "rec_clas_005197.evio.00405-00409.hipo";
        if(args.length>0) file = args[0];
        Level3Converter_MultiClass.convertFile(file, file+"_daq.h5");
        //Level3Converter_MultiClass.analyzer(file+"_daq.h5");*/

        String dir="/Users/tyson/data_repo/trigger_data/rgd/018437/";//rga
        String base="rec_clas_018437.evio.";//005197
        for (int file=0;file<10;file+=5){
    
            String fileS=String.valueOf(file);
            String fileS2=String.valueOf(file+4);

            String zeros="0000";
            String zeros2="0000";
            if(file>9){zeros="000";}
            if((file+4)>9){zeros2="000";}
            String fName=dir+base+zeros+fileS+"-"+zeros2+fileS2+".hipo";
            Level3Converter_MultiClass.convertFile(fName, dir+"daq_MC_"+fileS+".h5");

        }

    }
}
