/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.level3;

import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.io.HipoReader;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import twig.data.DataGroup;
import twig.data.H1F;
import twig.data.H2F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Utils {

    //--------------------------------------------------------------
    // DC TDC bank is a composite bank: {{ 12,1, "bbsbil" , 4098}}
    // EC ADC bank is a composite bank: {{ 11,2, "bbsbifs", 4098}}
    //--------------------------------------------------------------
    public static void fillDC(INDArray dc, CompositeNode dcBank, int order){
        int   nrows = dcBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double dcIncrement = 1./6.0;
        for(int row = 0; row < nrows; row++){
            int sector = dcBank.getInt(0, row);
            int  layer = dcBank.getInt(1, row);
            int   wire = dcBank.getInt(2, row);
            
            index[0]   = order*6 + (sector-1);
            //index[1]   = (layer-1)/6;
            index[1]   = 0;
            index[2]   = (layer-1)/6;
            index[3]   = wire - 1;


            //System.out.println(Arrays.toString(index));
            //System.out.println(Arrays.toString(index));
            double previous = dc.getDouble(index);
            dc.putScalar(index, previous + dcIncrement);
        }
    }
    
    public static void fillLabels(INDArray labels, CompositeNode trigger, int order){
        for(int r = 0; r < 6; r++){
            int value = trigger.getInt(0, r+1);
            //System.out.printf(" trigger %2d (%d) - sector = %2d trigger = %2d\n",
            //        trigger.getInt(0, 0), trigger.getRows(), r, trigger.getInt(0, r+1));
            if(value>0){
                labels.putScalar(new int[]{order*6+r,0} , 0.0);
                labels.putScalar(new int[]{order*6+r,1} , 1.0);
            } else {
                labels.putScalar(new int[]{order*6+r,0} , 1.0);
                labels.putScalar(new int[]{order*6+r,1} , 0.0);
            }
        }
    }
    
    public static void fillEC(INDArray dc, CompositeNode ecBank, int order){
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double dcIncrement = 1./6.0;
        for(int row = 0; row < nrows; row++){
            int sector = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0){
                double energy = (ADC/10000.0)/1.5;
                if(energy>=0.0&&energy<1.0){
                    index[0]   = order*6 + (sector-1);                    
                    index[1]   = 0;
                    index[2]   = (layer-1);
                    index[3]   = strip-1;
                    if(layer>6){
                        index[1] = layer - 3 - 1;
                        index[2] = strip + 36;
                    }
                    if(index[2]<72&&index[1]<6)
                        dc.putScalar(index, energy);
                }
            }
        }
    }
    
    public static INDArray[] createData(List<Event> events){
        CompositeNode nDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        int size = events.size();
        INDArray  DCArray = Nd4j.zeros( size*6 , 1, 6, 112);
    	INDArray  ECArray = Nd4j.zeros( size*6 , 1, 6,  72);
        for(int order = 0; order < size; order++){
            Level3Utils.fillDC(DCArray, nDC, order);
            Level3Utils.fillDC(ECArray, nEC, order);
        }
        return new INDArray[]{DCArray,ECArray};
    }
    
    public static List<H2F> getHist(INDArray dc, INDArray ec, int order){
        H2F hdc = new H2F("HDC", 112,0.5, 112.5, 6, 0.5, 6.5);
        H2F hec = new H2F("HEC",  72,0.5,  72.5, 6, 0.5, 6.5);
        for(int w = 0; w < 112; w++)
            for(int l = 0; l < 6; l++){
                double v = dc.getDouble(order,l,w);
                hdc.setBinContent(w, l, v);
            }
        
        for(int s = 0; s < 72; s++)
            for(int l = 0; l < 6; l++){
                double v = ec.getDouble(order,l,s);
                
                hec.setBinContent(s, l, v);
            }
        return Arrays.asList(hdc,hec);
    }
    
    public static DataGroup getGroup(INDArray dc, INDArray ec, int... order){
        DataGroup group = new DataGroup(2,order.length);
        for(int o = 0; o < order.length; o++){
            List<H2F> hL = Level3Utils.getHist(dc, ec, order[o]);
            group.add(hL.get(0),o*2, "");
            group.add(hL.get(1),o*2+1, "");
        }
        return group;
    }
    
    public static DataGroup getGroup(Event event){
        CompositeNode nDC = new CompositeNode(12,1,"bbsbil",4000);
        CompositeNode nEC = new CompositeNode(11,1,"bbsbil",4000);
        CompositeNode nRC = new CompositeNode( 5,1,"b",5);
        
        event.read(nDC, 12, 1);
        event.read(nEC, 11, 2);
        
        //nDC.print();
        //nEC.print();
        INDArray DCArray=Nd4j.zeros( 6 , 6, 112, 1);
    	INDArray ECArray=Nd4j.zeros( 6 , 6,  72, 1);
        Level3Utils.fillDC(DCArray, nDC, 0);
        Level3Utils.fillEC(ECArray, nEC, 0);
        DataGroup group = Level3Utils.getGroup(DCArray, ECArray, 0,1,2,3,4,5);
        
        event.read(nRC,5,1);
        for(int i = 1; i < nRC.getRows(); i++){
            int value = nRC.getInt(0, i);
            if(value>0) group.getData().get( (i-1)*2).attr().setTitleX("TRIGGER");
        }
        return group;
    }
    
    public static void main(String[] args){
        //String file = "/Users/gavalian/Work/Software/project-10.7/distribution/caos/coda/decoder/output.h5";
        String file = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.evio.00090-00094.hipo_daq.h5";
        HipoReader r = new HipoReader(file);
        Event e = new Event();
        TGCanvas c = new TGCanvas(1200,1300);        
        for(int i = 0; i < 5000; i++){
            r.getEvent(e, i+10);
            DataGroup grp = Level3Utils.getGroup(e);            
            grp.draw(c.view(), true);
            
            try {
                Thread.sleep(3000);
            } catch (InterruptedException ex) {
                Logger.getLogger(Level3Utils.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
}
