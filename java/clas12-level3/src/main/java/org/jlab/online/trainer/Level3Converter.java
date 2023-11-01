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


import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Converter {
    
    public static void analyzer(String file){
        
        CompositeNode nodeRC = new CompositeNode(  5, 1,  "b", 25);
        CompositeNode nodeET = new CompositeNode(  5, 2,  "b", 25);
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
    
    public static int[] convertTriggerLong(long bits) {
        int[] trigger = new int[21];

        //System.out.printf("%X - %X\n", bits, bits & 0xF);
        for (int i = 0; i < trigger.length; i++) {
            trigger[i] = 0;
            if (((bits >> i) & (1L)) != 0L)
                trigger[i] = 1;
            // System.out.println(Arrays.toString(trigger));
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
                if(Level3Converter.contains(sectors, sector)==true){
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
    
    public static void convertDC(Bank bank, CompositeNode node, int... sectors){
        node.setRows(0);
        int nrowsdc = bank.getRows();
        if(bank.getRows()<4000){

            //node.setRows(nrowsdc);
            int counter = 0;
            for(int row = 0; row < nrowsdc; row++){
                int sector = bank.getInt("sector", row);
                if(Level3Converter.contains(sectors, sector)==true){
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
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        CompositeNode nodeRC = new CompositeNode(  5, 1,  "b", 10);
        CompositeNode nodeET = new CompositeNode(  5, 2,  "b", 10);
        
        while(r.hasNext()){
            
            r.nextEvent(e);
            e.read(banks);
            
            e.read(dsts);

            int pid = dsts[0].getInt(0, 0);
            int status = dsts[0].getInt("status", 0);
            int partSector = 0;

            //only want particles in FD
            if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
                // always taking first particle from event ie pIndex 0
                int trIndex = Level3Converter.getTrackIndex(dsts[1], 0);
                // trIndex==-1 means that pIndex not found in track bank
                if (trIndex != -1) partSector = dsts[1].getInt("sector", trIndex);

                //only want particles associated to a sector
                if (partSector != 0) {
                    long bits = banks[2].getLong("trigger", 0);

                    int[] trigger = Level3Converter.convertTriggerLong(bits);

                    int trigSector = Level3Converter.getTriggerSector(trigger);

                    // if(trigger[0]>0) System.out.printf("%6d %4d %4d, %s\n",pid, trigSector,
                    // partSector ,Arrays.toString(trigger));

                    int[] labels = new int[] { pid, trigSector, partSector };

                    // want to get DC and EC from the particle sector
                    // not trigger sector
                    Level3Converter.convertDC(banks[0], nodeDC, partSector);
                    Level3Converter.convertEC(banks[1], nodeEC, partSector);

                    // nodeDC.print();
                    // nodeEC.print();

                    Node tnode = new Node(5, 4, labels);

                    if (nodeEC.getRows() > 0 && nodeDC.getRows() > 0) {
                        e.write(nodeEC);
                        e.write(nodeDC);
                        e.write(tnode);
                        w.addEvent(e);
                    }
                }
            }

            /*
            if(pid==11&&Math.abs(status)>=2000&&Math.abs(status)<3000){
        
                int[] electron = new int[]{0,0,0,0,0,0,0};
                
                int pindex = dsts[1].getInt("pindex", 0);
                int sector = dsts[1].getInt("sector", 0);
                
                if(pindex==0) { electron[0] = 1; electron[sector] = 1;}
                
                nodeEC.setRows(0);
                nodeDC.setRows(0);
                nodeRC.setRows(0);
                nodeET.setRows(0);
                
                int nrowsec = banks[1].getRows();
                
               // if(banks[2].getRows()>0){
               //     int[] trigger = new int[7];
               //     long     bits = banks[2].getLong("trigger", 0);
               //     for(int i = 0; i < trigger.length; i++) {
               //        trigger[i] = 0;
               //         if( ((bits>>i)&(1L)) != 0L) trigger[i] = 1;
               //         //System.out.println(Arrays.toString(trigger));
               //     }
               //     nodeRC.setRows(7); nodeET.setRows(7);
               //     for(int i = 0; i < 7; i++) {
               //         nodeRC.putByte(0, i, (byte) trigger[i]);
               //         nodeET.putByte(0, i, (byte) electron[i]);
               //     }
               //}
                if(nrowsec<4000){
                    nodeEC.setRows(nrowsec);
                    for(int row = 0; row < nrowsec; row++){
                        nodeEC.putByte(  0, row, (byte) banks[1].getInt("sector", row));
                        nodeEC.putByte(  1, row, (byte) banks[1].getInt("layer", row));
                        nodeEC.putShort( 2, row, (short) banks[1].getInt("component", row));
                        nodeEC.putByte(  3, row, (byte) banks[1].getInt("order", row));
                        nodeEC.putInt(   4, row,  banks[1].getInt("ADC", row));
                    }
                } 
                
                int nrowsdc = banks[0].getRows();
                if(nrowsdc<4000){
                    nodeDC.setRows(nrowsdc);
                    for(int row = 0; row < nrowsdc; row++){
                        nodeDC.putByte(  0, row, (byte) banks[0].getInt("sector", row));
                        nodeDC.putByte(  1, row, (byte) banks[0].getInt("layer", row));
                        nodeDC.putShort( 2, row, (short) banks[0].getInt("component", row));
                        nodeDC.putByte(  3, row, (byte) banks[0].getInt("order", row));
                        nodeDC.putInt(   4, row,  banks[0].getInt("TDC", row));
                    }
                }
                if(nodeEC.getRows()>0) e.write(nodeEC);
                if(nodeDC.getRows()>0) e.write(nodeDC);
                if(nodeRC.getRows()>0) e.write(nodeRC);
                if(nodeET.getRows()>0) e.write(nodeET);
                //int trigger = nodeRC.getInt(0, 0);
                //if(trigger>0)
                w.addEvent(e);
            }*/
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
        Level3Converter.convertFile(file, file+"_daq.h5");
        //Level3Converter.analyzer(file+"_daq.h5");*/

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
            Level3Converter.convertFile(fName, dir+"daq_"+fileS+".h5");

        }

    }
}
