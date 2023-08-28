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

/**
 *
 * @author gavalian
 */
public class Level3Converter {
    public static void convertFile(String file, String output){
        HipoReader r = new HipoReader(file);
        HipoWriter w = HipoWriter.create(output, r);
        Event e = new Event();
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        CompositeNode nodeRC = new CompositeNode(  5, 1,  "b", 10);
        CompositeNode nodeET = new CompositeNode(  5, 2,  "b", 10);
        
        while(r.hasNext()){
            
            r.nextEvent(e);
            e.read(banks);

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
                nodeRC.setRows(7);
                for(int i = 0; i < 7; i++) nodeRC.putByte(0, i, (byte)trigger[i]);
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
            int trigger = nodeRC.getInt(0, 0);
            if(trigger>0)
                w.addEvent(e);
        }
        
        w.close();
    }
    
    public static void main(String[] args){
        //String file = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.evio.00090-00094.hipo";
        String file = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.h5_000001";

        Level3Converter.convertFile(file, file+"_daq.h5");
    }
}
