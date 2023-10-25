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
import java.lang.Math;

import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian, tyson
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
    
    public static double convertEC(Bank bank, CompositeNode node, int... sectors){
        double totEnergy=0;
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
                    totEnergy+=(double) (bank.getInt("ADC", row)/10000.0)/1.5;
                    counter++;
                }
            }
        }
        return totEnergy;
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

    public static double getPartE(Bank PartBank, int pIndex,double massPart){
        double px=PartBank.getFloat("px", pIndex);
        double py=PartBank.getFloat("py", pIndex);
        double pz=PartBank.getFloat("pz", pIndex);
        double p=Math.sqrt(px*px+py*py+pz*pz);
        double energy=Math.sqrt(p*p+massPart*massPart);
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

    public static List<Integer> getPIndices_fSector_fCal(Bank CalBank, int sector){
        List<Integer> pIndices = new ArrayList<Integer>();
        for (int row=0; row< CalBank.getRows();row++){
            int calPIndex=CalBank.getInt("pindex", row);
            int calSector=CalBank.getInt("sector", row);
            if (calSector==sector){pIndices.add(calPIndex);}
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

    public static int getPID_inSector(List<Integer> pIndices,Bank PartBank) {
        // pid=-1 denotes no track in sector
        int pid = -1;
        // check if there's at least 1 track in sector
        if (pIndices.size() > 0) {
            pid = PartBank.getInt(0, 0);
            // loop over particle indices in sector to check if there's an electron in
            // sector
            for (int i = 0; i < pIndices.size(); i++) {
                if (PartBank.getInt(0, i) == 11) {
                    pid = 11;
                }
                // if there's already an electron we don't want
                // to overwrite it
                // if not check if there's a pi-
                if (pid != 11 && PartBank.getInt(0, i) == -211) {
                    pid = -211;
                }
            }
        }
        return pid;
    }

    public static double getTotal_neutralE_inSector(List<Integer> pIndices_fCal, Bank PartBank) {
        double nEnergy = 0;
        // loop over particle indices in cal in sector to check if there's neutrals
        for (int i = 0; i < pIndices_fCal.size(); i++) {
            // check if we have neutral pids
            if (PartBank.getInt(0, i) == 22) {
                // nEdep+=Level3Converter_MultiClass.getCALE(dsts[2],i);
                nEnergy += getPartE(PartBank, i, 0);
            } else if (PartBank.getInt(0, i) == 2112) {
                // nEdep+=Level3Converter_MultiClass.getCALE(dsts[2],i);
                nEnergy += getPartE(PartBank, i, 0.93957);
            }
        }
        return nEnergy;
    }
    
    public static void convertFile(String file, String output){
        HipoReader r = new HipoReader(file);
        HipoWriter w = HipoWriter.create(output, r);
        Event e = new Event();
        Event e_out = new Event();
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);

        int nTag1=0,nTag2=0,nTag3=0,nTag4=0,nTag5=0,nTag6=0,nTag7=0;
        
        while(r.hasNext()){
            
            r.nextEvent(e);
            e.read(banks);
            
            e.read(dsts);

            //loop over sectors
            for (int sect = 1; sect < 7; sect++) {
                // get a list of pIndices in the sector associated with tracks
                List<Integer> pIndices = Level3Converter_MultiClass.getPIndices_fSector(dsts[1], sect);
                //find if there's an electron, pion, other or no track in the track pindices in sector
                int pid=getPID_inSector(pIndices,dsts[0]);

                // get a list of pIndices in the sector associated with calorimeter
                List<Integer> pIndices_fCal = Level3Converter_MultiClass.getPIndices_fSector_fCal(dsts[2], sect);
                //if there's neutrals in cal pindices in sector, get their total energy
                double nEnergy=getTotal_neutralE_inSector(pIndices_fCal,dsts[0]);

                // get DC and EC for the sector
                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                double nEdep=Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);

                /*if(nEnergy>0){
                    System.out.printf("per event Edep: (%f)\n",nEnergy);
                } */

                //tag 0 is trash
                int tag=0;

                //there's one or more photon 
                //with >1 GeV E dep in calorimeters in total
                if (nEnergy>1) {
                    // tag=1 means there's an electron
                    if (pid == 11) {
                        tag = 1;
                        nTag1++;
                    } // tag 5 means there's a pion and neutral
                    else if (pid == -211) {
                        tag = 5;
                        nTag5++;
                    } // tag 7 means there's no track and a neutral
                    else if (pid == -1) {
                        tag = 7;
                        nTag7++;
                    } // tag 6 means there's another particle (not el not pion) and neutral
                    else {
                        tag = 6;
                        nTag6++;
                    }
                } else {
                    // tag=1 means there's an electron
                    if (pid == 11) {
                        tag = 1;
                        nTag1++;
                    } // tag 2 means there's a negative pion but no neutral
                    else if (pid == -211) {
                        tag = 2;
                        nTag2++;
                    } // tag 4 means there's no track and no neutral
                    else if (pid == -1 ) {
                        tag = 4;
                        nTag4++;
                    } // tag 3 means there's another particle (not el or pi-) but no neutral
                    else {
                        tag = 3;
                        nTag3++;
                    }
                }

                //for each data entry we could record trigger
                //could help with some logic in training data
                //eg requiring conventional trigger to be wrong
                /*long bits = banks[2].getLong("trigger", 0);
                int[] trigger = Level3Converter_MultiClass.convertTriggerLong(bits);
                int trigSector = Level3Converter_MultiClass.getTriggerSector(trigger);*/

                int[] labels = new int[] { pid, tag, sect };
                
                // nodeDC.print();
                // nodeEC.print();

                //System.out.printf("Tag: (%d)",tag);

                Node tnode = new Node(5, 4, labels);

                if (nodeEC.getRows() > 0 && nodeDC.getRows() > 0 && tag>0) {
                    
                    e_out.reset();
                    e_out.write(nodeEC);
                    e_out.write(nodeDC);
                    e_out.write(tnode);

                    e_out.setEventTag(tag);

                    w.addEvent(e_out);
                }


            }
        }
        w.close();

        System.out.println("Number of events from:");
        System.out.printf("Tags 1 (%d), 2 (%d), 3 (%d)\n",nTag1,nTag2,nTag3);
        System.out.printf("Tags 4 (%d), 5 (%d), 6 (%d), 7 (%d)\n",nTag4,nTag5,nTag6,nTag7);
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
        

        /*String dir="/Users/tyson/data_repo/trigger_data/rga/";
        String base="rec_clas_005197.evio.";*/

        String dir="/Users/tyson/data_repo/trigger_data/rgd/018437/";
        String base="rec_clas_018437.evio.";

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
