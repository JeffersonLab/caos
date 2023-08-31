/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;
import java.util.Arrays;
import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Converter {
    
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
                
                if(banks[2].getRows()>0){
                    int[] trigger = new int[7];
                    long     bits = banks[2].getLong("trigger", 0);
                    for(int i = 0; i < trigger.length; i++) {
                        trigger[i] = 0;
                        if( ((bits>>i)&(1L)) != 0L) trigger[i] = 1;
                        //System.out.println(Arrays.toString(trigger));
                    }
                    nodeRC.setRows(7); nodeET.setRows(7);
                    for(int i = 0; i < 7; i++) {
                        nodeRC.putByte(0, i, (byte) trigger[i]);
                        nodeET.putByte(0, i, (byte) electron[i]);
                    }
                }
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
                int trigger = nodeRC.getInt(0, 0);
                if(trigger>0)
                    w.addEvent(e);
            }
        }
        
        w.close();
    }
    
    public static void main(String[] args){
        //String file = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.evio.00090-00094.hipo";
        String file = "/Users/gavalian/Work/DataSpace/trigger/recon/006152/rec_clas_006152.evio.00055-00059.hipo";

        //Level3Converter.convertFile(file, file+"_daq.h5");
        Level3Converter.analyzer(file+"_daq.h5");
    }
}
