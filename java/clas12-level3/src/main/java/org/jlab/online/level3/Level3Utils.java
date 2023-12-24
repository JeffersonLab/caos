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
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import twig.data.DataGroup;
import twig.data.H1F;
import twig.data.H2F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Utils {

    
    public static void fillDC(INDArray dc, CompositeNode dcBank, int sector, int order){
        int   nrows = dcBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double dcIncrement = 1./6.0;
        for(int row = 0; row < nrows; row++){
            int  sect = dcBank.getInt(0, row);
            int  layer = dcBank.getInt(1, row);
            int   wire = dcBank.getInt(2, row);
            
            index[0]   = order;
            index[1]   = 0;
            index[2]   = (layer-1)/6;
            index[3]   = wire - 1;


            //System.out.println(Arrays.toString(index));
            //System.out.println(Arrays.toString(index));
            if(sect==sector){
                double previous = dc.getDouble(index);
                dc.putScalar(index, previous + dcIncrement);
            }
        }
    }

    public static void fillDC_SepSL(INDArray dc, CompositeNode dcBank, int sector, int order){
        int   nrows = dcBank.getRows();
        int[] index = new int[]{0,0,0,0};
        for(int row = 0; row < nrows; row++){
            int  sect = dcBank.getInt(0, row);
            int  layer = dcBank.getInt(1, row);
            int   wire = dcBank.getInt(2, row);
            
            index[0]   = order;
            index[1]   = (layer-1)/6; //superlayer
            index[2]   = (layer-1)%6; //layer
            index[3]   = wire - 1;


            //System.out.println(Arrays.toString(index));
            //System.out.println(Arrays.toString(index));
            if(sect==sector){
                dc.putScalar(index, 1);
            }
        }
    }

    public static void fillDC_wLayers(INDArray dc, CompositeNode dcBank, int sector, int order){
        int   nrows = dcBank.getRows();
        int[] index = new int[]{0,0,0,0};
        
        for(int row = 0; row < nrows; row++){
            int  sect = dcBank.getInt(0, row);
            int  layer = dcBank.getInt(1, row);
            int   wire = dcBank.getInt(2, row);
            
            index[0]   = order;
            index[1]   = 0;
            index[2]   = (layer-1);
            index[3]   = wire - 1;


            //System.out.println(Arrays.toString(index));
            //System.out.println(Arrays.toString(index));
            if(sect==sector){
                dc.putScalar(index, 1);
            }
        }
    }

    public static void fillDC_wLayersTDC(INDArray dc, CompositeNode dcBank, int sector, int order){
        int   nrows = dcBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double tdc_min=1;
        double tdc_max=1500;
        
        for(int row = 0; row < nrows; row++){
            int  sect = dcBank.getInt(0, row);
            int  layer = dcBank.getInt(1, row);
            int   wire = dcBank.getInt(2, row);
            double   tdc = (dcBank.getInt(4, row)-tdc_min)/tdc_max;
            
            index[0]   = order;
            index[1]   = 0;
            index[2]   = (layer-1);
            index[3]   = wire - 1;


            //System.out.println(Arrays.toString(index));
            //System.out.println(Arrays.toString(index));
            if(sect==sector && tdc>0.00 &&tdc<1.00){
                dc.putScalar(index, tdc);
            }
        }
    }
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
   
    public static void fillLabels(INDArray labels, int label, int order){
        if(label==0){
            labels.putScalar(new int[]{order,0} , 1.0);
            labels.putScalar(new int[]{order,1} , 0.0);
        } else {
            labels.putScalar(new int[]{order,0} , 0.0);
            labels.putScalar(new int[]{order,1} , 1.0);
        }        
    }

    public static void fillLabels_MultiClass(INDArray labels,int tags_size, int tag_index, int order){
        //loop over possible tags
        for (int i=0;i<tags_size;i++){
            double val=0.0;
            //if event at order correspond to desired tag
            //then label it with 1
            if(i==tag_index){val=1.0;}
            //put val at index i in output array
            labels.putScalar(new int[]{order,i} , val);
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
                double energy = (ADC/10000.0)/1.5/3.0;
                if(energy>=0.0&&energy<1.0){
                    index[0]   = order*6 + (sector-1);                    
                    index[1]   = 0;
                    index[2]   = (layer-1);
                    index[3]   = strip-1;
                    if(layer>6){
                        index[2] = layer - 3 - 1;
                        index[3] = (strip + 36)-1;
                    }
                    if(index[3]<72&&index[2]<6)
                        dc.putScalar(index, energy);
                }
            }
        }
    }

    public static void fillHTCC(INDArray htcc, CompositeNode htccBank, int sector , int order){
        int   nrows = htccBank.getRows();
        int[] index = new int[]{0,0,0,0};

        for(int row = 0; row < nrows; row++){
            
            int   sect = htccBank.getInt(0, row);
            int  layer = htccBank.getInt(1, row); //1 or 2
            int  component = htccBank.getInt(2, row); //1-4
            int    ADC = htccBank.getInt(4, row);

            if(ADC>0){
                double energy = (ADC/5000.0); //&&energy<1.0
 
                if(energy>=0.0&&sect==sector){
                    index[0]   = order;
                    //index[1] is channel number set to 1
                    index[2]   = ((layer-1)*4+component)-1;
                    
                    if(index[1]<8){
                        htcc.putScalar(index, energy);
                    }
                }
            }
        }
    }

    public static void fillFTOF(INDArray ftof, CompositeNode ftofBank, int sector , int order){
        int   nrows = ftofBank.getRows();
        int[] index = new int[]{0,0,0,0};

        for(int row = 0; row < nrows; row++){
            
            int   sect = ftofBank.getInt(0, row);
            int  layer = ftofBank.getInt(1, row); //1 or 2
            int  component = ftofBank.getInt(2, row); //1-4
            int    ADC = ftofBank.getInt(4, row);

            if(ADC>0 && layer==2){//only keep layer 1-b for now
                double energy = (ADC/50000.0); 
 
                if(energy>=0.0&&sect==sector){//&&energy<1.0
                    index[0]   = order;
                    //index 1 is channel number set to 1
                    index[2]   = component-1;
                    
                    if(index[1]<62){
                     ftof.putScalar(index, energy);
                    }
                }
            }
        }
    }

    public static void fillFTOF_wNorm(INDArray ftof, CompositeNode ftofBank, int sector , int order){
        int   nrows = ftofBank.getRows();
        int[] index = new int[]{0,0,0,0};

        for(int row = 0; row < nrows; row++){
            
            int   sect = ftofBank.getInt(0, row);
            int  layer = ftofBank.getInt(1, row); //1 or 2
            int  component = ftofBank.getInt(2, row); //1-4
            int    ADC = ftofBank.getInt(4, row);

            if(ADC>0 && layer==2){//only keep layer 1-b for now
                double energy = (ADC/50000.0); 
 
                if(energy>=0.0&&sect==sector){//&&energy<1.0
                    index[0]   = order;
                    //index 1 is channel number set to 0
                    index[2]   = component-1;
                    
                    if(index[2]<62 && energy>0.015){
                     ftof.putScalar(index, 1);
                    }
                }
            }
        }
    }

    public static int fillEC(INDArray dc, CompositeNode ecBank, int sector , int order){
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double dcIncrement = 1./6.0;
        int nHits=0;

        for(int row = 0; row < nrows; row++){
            
            int   sect = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0){
                double energy = (ADC/10000.0)/1.5/3.0; 
 
                //----------------
                if(energy>=0.0&&energy<1.0&&sect==sector){
                    index[0]   = order;                    
                    index[1]   = 0;
                    index[2]   = (layer-1);
                    index[3]   = strip-1;
                    if(layer>6){
                        index[2] = layer - 3 - 1;
                        index[3] = (strip + 36)-1;
                    }
                    
                    
                    if(index[3]<72&&index[2]<6){
                        nHits++;
                        dc.putScalar(index, energy);
                    }
                }
                //----------------
                
            }
        }
        return nHits;
    }

    public static void fillEC_noNorm(INDArray dc, CompositeNode ecBank, int sector , int order){
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0,0,0};

        for(int row = 0; row < nrows; row++){
            
            int   sect = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0){
                double energy = ADC; 
 
                //----------------
                if(energy>=0.0 && sect==sector){
                    index[0]   = order;                    
                    index[1]   = 0;
                    index[2]   = (layer-1);
                    index[3]   = strip-1;
                    if(layer>6){
                        index[2] = layer - 3 - 1;
                        index[3] = (strip + 36)-1;
                    }
                    
                    
                    if(index[3]<72&&index[2]<6){
                        dc.putScalar(index, energy);
                    }
                }
                //----------------
                
            }
        }
    }

    public static void fillECin(INDArray ec, CompositeNode ecBank, int sector , int order){
        
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0,0,0};

        for(int row = 0; row < nrows; row++){
            
            int   sect = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0.0){
                double energy = (ADC/10000.0)/1.5/3.0;
                //----------------
                if(sect==sector){

                    if (layer > 3 && layer < 7) {
                        index[0] = order;
                        index[3] = ((layer-4)*36+strip)-1;
                        //index 1 is channels, 0 channels
                        //index 2 would be layers but only 1 (ecin)

                        if (index[3] < 108) {
                            ec.putScalar(index, energy);
                        }
                    }
                }
                //----------------  
            }
        }
    }

    public static void fillLabels_ClusterFinder(INDArray ec, CompositeNode ecBank, int sector , int order){
        
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0};

        double ADC_max_U=0,ADC_max_V=0,ADC_max_W=0;

        List<Integer> js= new ArrayList<>();

        for(int row = 0; row < nrows; row++){
            
            int   sect = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0.0){
                
                //----------------
                if(sect==sector){

                    if (layer > 3 && layer < 7) {
                        index[0] = order;
                        index[1] = ((layer-4)*36+strip)-1;

                        if (index[1] < 108) {
                            ec.putScalar(index, ADC);
                            js.add(index[1]);
                            if(layer==4){if(ADC>ADC_max_U){ADC_max_U=ADC;}}
                            if(layer==5){if(ADC>ADC_max_V){ADC_max_V=ADC;}}
                            if(layer==6){if(ADC>ADC_max_W){ADC_max_W=ADC;}}
                        }
                    }
                }
                //----------------  
            }
        }

        for (int j :js){
            /*if(j<36){ec.putScalar(new int[]{order,j},ec.getFloat(order,j)/ADC_max_U);}
            else if(j>35 && j<72){ec.putScalar(new int[]{order,j},ec.getFloat(order,j)/ADC_max_V);}
            else if(j>71){ec.putScalar(new int[]{order,j},ec.getFloat(order,j)/ADC_max_W);}*/
            if(j<36){
                if(ec.getFloat(order,j)==ADC_max_U){
                    ec.putScalar(new int[]{order,j},1);
                } else{
                    ec.putScalar(new int[]{order,j},0);
                }
            }
            else if(j>35 && j<72){
                if(ec.getFloat(order,j)==ADC_max_V){
                    ec.putScalar(new int[]{order,j},1);
                } else{
                    ec.putScalar(new int[]{order,j},0);
                }
            }
            else if(j>71){
                if(ec.getFloat(order,j)==ADC_max_W){
                    ec.putScalar(new int[]{order,j},1);
                } else{
                    ec.putScalar(new int[]{order,j},0);
                }
            }
        }
    }
    
    //energies just for testing
    public static void fillEC(INDArray dc, CompositeNode ecBank, int sector , int order,List<Double> energies){
        int   nrows = ecBank.getRows();
        int[] index = new int[]{0,0,0,0};
        double dcIncrement = 1./6.0;

        for(int row = 0; row < nrows; row++){
            
            int   sect = ecBank.getInt(0, row);
            int  layer = ecBank.getInt(1, row);
            int  strip = ecBank.getInt(2, row);
            int    ADC = ecBank.getInt(4, row);

            if(ADC>0){
                double energy = (ADC/10000.0)/1.5/3.0;
                //System.out.printf("energy in EC (%f)\n", energy);              
                energies.add(energy); //for testing
 
                //----------------
                //=0.0004, 
                if(energy>0.0&&energy<1.0&&sect==sector){
                    
                    index[0]   = order;                    
                    index[1]   = 0;
                    index[2]   = (layer-1);
                    index[3]   = strip-1;
                    if(layer>6){
                        index[2] = layer - 3 - 1;
                        index[3] = (strip + 36)-1;
                    }
                    
                    
                    if(index[3]<72&&index[2]<6)
                        dc.putScalar(index, energy);
                }
                //----------------
                
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
            events.get(order).read(nDC,12,1);
            events.get(order).read(nEC,11,2);
            Level3Utils.fillDC(DCArray, nDC, order);
            Level3Utils.fillEC(ECArray, nEC, order);
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
