/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.data.base.DataFrame;
import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.io.File;
import java.io.IOException;
import java.lang.Math;

import twig.data.GraphErrors;
import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author tyson
 */
public class Level3Tester {

    ComputationGraph network = null;

    public Level3Tester() {

    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Tester.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
   
    public static MultiDataSet getData(String file,int max,double beamE){

        HipoReader r = new HipoReader(file);
        Event e = new Event();

        int nMax = max;
        if (r.entries() < max)
            nMax = r.entries();

        INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
        INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
        INDArray OUTArray = Nd4j.zeros(nMax, 4);
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        
        int counter=0,nEls=0,nL1=0,nOther=0,nEmpty=0;
        while(r.hasNext() && counter<(nMax-6)){
            
            r.nextEvent(e);
            e.read(banks);
            e.read(dsts);

            long bits = banks[2].getLong("trigger", 0);
            int[] trigger = Level3Converter_MultiClass.convertTriggerLong(bits);
            //System.out.println(Arrays.toString(trigger));

            //list of helpful arrays
            List<Integer> elIndexes = new ArrayList<Integer>();
            List<Integer> elSectors = new ArrayList<Integer>();
            Map<Integer, Double> q2ByIndex=new HashMap<Integer, Double>();
            Map<Integer, Double> q2BySector=new HashMap<Integer, Double>();
            List<Integer> otherIndexes = new ArrayList<Integer>();
            List<Integer> otherSectors = new ArrayList<Integer>();
            Map<Integer, Double> otherpByIndex = new HashMap<Integer, Double>();
            Map<Integer, Double> otherpBySector = new HashMap<Integer, Double>();

            ////get part pIndex and p
            for (int i = 0; i < dsts[0].getRows(); i++) {
                int pid = dsts[0].getInt("pid", i);
                int status = dsts[0].getInt("status", i);
                if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
                    if (pid == 11) {
                        elIndexes.add(i);
                        double[] pthetaphi=Level3Converter_MultiClass.calcPThetaPhi(dsts[0], i);
                        double q2=2*beamE*pthetaphi[0]*(1-Math.cos(pthetaphi[1]));
                        q2ByIndex.put(i, q2);
                    } else if(pid!=2112 && pid!=22 && pid!=0 && pid!=11){
                        otherIndexes.add(i);
                        double[] pthetaphi=Level3Converter_MultiClass.calcPThetaPhi(dsts[0], i);
                        otherpByIndex.put(i,pthetaphi[0]);
                    }
                }
            }
            //get part sector and put part p by sector
            for (int k = 0; k < dsts[1].getRows(); k++) {
                int pindex = dsts[1].getInt("pindex", k);
                int sectorTrk = dsts[1].getInt("sector", k);
                if (elIndexes.contains(pindex)) {
                    elSectors.add(sectorTrk);
                    q2BySector.put(sectorTrk, q2ByIndex.get(pindex));
                }  else if (otherIndexes.contains(pindex)) {
                    otherSectors.add(sectorTrk);
                    otherpBySector.put(sectorTrk, otherpByIndex.get(pindex));
                }
            }

            //loop over sectors
            for (int sect = 1; sect < 7; sect++) {
  
                int gotEl = 0;
                double q2=0;
                if(elSectors.contains(sect)){
                    gotEl=1;
                    q2=q2BySector.get(sect);
                } else if(otherSectors.contains(sect)){
                    gotEl=2;
                    q2=otherpBySector.get(sect);
                }

                // get DC and EC for the sector
                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                double nEdep = Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);


                if (nodeEC.getRows() > 0 && nodeDC.getRows() > 0 ) { //&& trigger[0]>1  //&& trigger[31]==1
                
                    Level3Utils.fillDC(DCArray, nodeDC, sect, counter);
                    Level3Utils.fillEC(ECArray, nodeEC, sect, counter);

                    INDArray EventDCArray=DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());
                    INDArray EventECArray=ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());

                    long hasL1 = Level3Converter_MultiClass.isTriggerInSector(trigger, sect);
                    if(hasL1>0){nL1++;}

                    // && hasL1==1
                    if (EventDCArray.any() && EventECArray.any() ) { // check that the images aren't all empty

                         
                        if(gotEl==1){nEls++;}
                        if(gotEl==2){nOther++;}
                        if(gotEl==0){nEmpty++;}
                        
                        OUTArray.putScalar(new int[] { counter, 0 }, gotEl);
                        OUTArray.putScalar(new int[] { counter, 1 }, hasL1);
                        OUTArray.putScalar(new int[]{counter,2},q2);
                        OUTArray.putScalar(new int[]{counter,3},sect);
                        counter++;
                    } else {
                        //erase last entry
                        DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 112));
                        ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 72));
                    }
                }

            }
        }
        //System.out.print(OUTArray);
        //System.out.print(DCArray);
        //System.out.print(ECArray);
        System.out.printf("counter %d, nEl %d, nOther %d,nEmpty %d, nL1 %d\n\n",counter,nEls,nOther,nEmpty,nL1);


        INDArray[] inputs = new INDArray[2];
        INDArray[] outputs = new INDArray[1];
        inputs[0]=DCArray;
        inputs[1]=ECArray;
        outputs[0]=OUTArray;
        MultiDataSet dataset = new MultiDataSet(inputs,outputs);
        //dataset.shuffle();
        return dataset;
    }

    public static void PlotResponse(INDArray output, INDArray Labels,int LabelVal,int elClass,String part) {
        long NEvents = output.shape()[0];
        H1F hRespPos = new H1F(part+" in Sector", 100, 0, 1);
        hRespPos.attr().setLineColor(2);
        hRespPos.attr().setFillColor(2);
        hRespPos.attr().setLineWidth(3);
        hRespPos.attr().setTitleX("Response");
        H1F hRespNeg = new H1F("No "+part+" in Sector", 100, 0, 1);
        hRespNeg.attr().setLineColor(5);
        hRespNeg.attr().setLineWidth(3);
        hRespNeg.attr().setTitleX("Response");
        //Sort predictions into those made on the positive/or negative samples
        for(long i=0;i<NEvents;i+=1) {
            if(Labels.getFloat(i,0)==LabelVal) {
            hRespPos.fill(output.getFloat(i,elClass));
            } else {
            hRespNeg.fill(output.getFloat(i,elClass));
            }
        }
    
        TGCanvas c = new TGCanvas();
        
        c.setTitle("Response");
        c.draw(hRespPos).draw(hRespNeg,"same");
        c.region().showLegend(0.05, 0.95);
            
        }//End of PlotResponse

    //Labels col 0 is 1 if there's an e-, 0 otherwise
    //Labels col 1 is 1 if l1 trigger fired, 0 otherwise
    //Labels col 2 is q2 when col 1 is 1, 0 otherwise
    public static INDArray getMetsForBin(INDArray outputs, INDArray Labels,int LabelVal,double thresh,int elClass,int cutVar,double low,double high){
        INDArray metrics = Nd4j.zeros(4,1);
        long nEvents = outputs.shape()[0];

        double TP=0,FN=0,FP=0;
        double TP_l1=0,FN_l1=0,FP_l1=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, cutVar) > low && Labels.getFloat(i,cutVar)<high) {
                if (Labels.getFloat(i, 0) == LabelVal) {
                    if (outputs.getFloat(i, elClass) > thresh) {
                        TP++;
                    } else {
                        FN++;
                    } // Check model prediction
                    if (Labels.getFloat(i, 1) == 1) {
                        TP_l1++;
                    } else {
                        FN_l1++;
                    }
                } else {
                    if (outputs.getFloat(i, elClass) > thresh) {
                        FP++;
                    }
                    if (Labels.getFloat(i, 1) == 1) {
                        FP_l1++;
                    }
                } // Check true label
            }
        }
	    double Pur=TP/(TP+FP);
	    double Eff=TP/(TP+FN);
	    metrics.putScalar(new int[] {0,0}, Pur);
	    metrics.putScalar(new int[] {1,0}, Eff);
        double Pur_l1=TP_l1/(TP_l1+FP_l1);
	    double Eff_l1=TP_l1/(TP_l1+FN_l1);
        metrics.putScalar(new int[] {2,0}, Pur_l1);
	    metrics.putScalar(new int[] {3,0}, Eff_l1);

        return metrics;
    }

    //Labels col 0 is 1 if there's an e-, 0 otherwise
    //Labels col 1 is 1 if l1 trigger fired, 0 otherwise
    public static INDArray getMetrics(INDArray outputs, INDArray Labels,int LabelVal,double thresh,int elClass){
        INDArray metrics = Nd4j.zeros(7,1);
        long nEvents = outputs.shape()[0];

        int nEls=0,nTrig=0;
        double TP=0,FP=0,FN=0;
        double TP_l1=0,FP_l1=0,FN_l1=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, 0) == LabelVal) {
                nEls++;
                if (outputs.getFloat(i, elClass) > thresh) {
                    TP++;
                } else {
                    FN++;
                } // Check model prediction
                if(Labels.getFloat(i, 1) == 1){
                    nTrig++;
                    TP_l1++;
                } else{
                    FN_l1++;
                }
            } else {
                if (outputs.getFloat(i, elClass) > thresh) {
                    FP++;
                } 
                if(Labels.getFloat(i, 1) == 1){
                    nTrig++;
                    FP_l1++;
                }
            } // Check true label
        }
	    double Pur=TP/(TP+FP);
	    double Eff=TP/(TP+FN);
	    metrics.putScalar(new int[] {0,0}, Pur);
	    metrics.putScalar(new int[] {1,0}, Eff);
        double Pur_l1=TP_l1/(TP_l1+FP_l1);
	    double Eff_l1=TP_l1/(TP_l1+FN_l1);
        metrics.putScalar(new int[] {2,0}, Pur_l1);
	    metrics.putScalar(new int[] {3,0}, Eff_l1);
        metrics.putScalar(new int[] {4,0}, TP);
	    metrics.putScalar(new int[] {5,0}, FP);
        metrics.putScalar(new int[] {6,0}, FN);

        /*System.out.printf("Theres %d electrons in sample\n", nEls);
        System.out.printf("L1 trigger fired %d times in sample\n", nTrig);*/
        return metrics;
    }

    public double findBestThreshold(MultiDataSet data,int elClass,double effLow,int LabelVal){
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        
        GraphErrors gEff = new GraphErrors();
        gEff.attr().setMarkerColor(2);
        gEff.attr().setMarkerSize(10);
        gEff.attr().setTitle("Efficiency");
        gEff.attr().setTitleX("Response");
        gEff.attr().setTitleY("Metrics");
        GraphErrors gPur = new GraphErrors();
        gPur.attr().setMarkerColor(5);
        gPur.attr().setMarkerSize(10);
        gPur.attr().setTitle("Purity");
        gPur.attr().setTitleX("Response");
        gPur.attr().setTitleY("Metrics");
        double bestRespTh = 0;
        double bestPuratEffLow= 0;

        // Loop over threshold on the response
        for (double RespTh = 0.01; RespTh < 0.99; RespTh += 0.01) {
            INDArray metrics=Level3Tester.getMetrics(outputs[0],data.getLabels()[0],LabelVal,RespTh, elClass);
            double Pur = metrics.getFloat(0, 0);
            double Eff = metrics.getFloat(1, 0);
            gPur.addPoint(RespTh, Pur, 0, 0);
            gEff.addPoint(RespTh, Eff, 0, 0);
            if (Eff > effLow) {
                if (Pur > bestPuratEffLow) {
                    bestPuratEffLow = Pur;
                    bestRespTh = RespTh;
                }
            }
        } // Increment threshold on response

        System.out.format("%n Best Purity at Efficiency above %f: %.3f at a threshold on the response of %.3f %n%n",
                effLow,bestPuratEffLow, bestRespTh);

        TGCanvas c = new TGCanvas();
        c.setTitle("Metrics vs Response");
        c.draw(gEff).draw(gPur, "same");
        c.region().showLegend(0.05, 0.95);

        return bestRespTh;
    }

    public static void plotVarDep(MultiDataSet data, INDArray outputs, double thresh, int elClass, int LabelVal,
            Boolean addPur, Boolean addL1, int cutVar, String varName, String varUnits,double low, double high,double step) {

        String yTitle="Metrics";
        if(!addPur){yTitle="Efficiency";}

        GraphErrors gEff = new GraphErrors();
        gEff.attr().setMarkerColor(2);
        gEff.attr().setMarkerSize(10);
        gEff.attr().setTitle("Level3 Efficiency");
        gEff.attr().setTitleX(varName+" "+varUnits);
        gEff.attr().setTitleY(yTitle);

        GraphErrors gEff_l1 = new GraphErrors();
        gEff_l1.attr().setMarkerColor(9);
        gEff_l1.attr().setMarkerSize(10);
        gEff_l1.attr().setTitle("Level1 Efficiency");
        gEff_l1.attr().setTitleX(varName+" "+varUnits);
        gEff_l1.attr().setTitleY(yTitle);

        GraphErrors gPur = new GraphErrors();
        gPur.attr().setMarkerColor(5);
        gPur.attr().setMarkerSize(10);
        gPur.attr().setTitle("Level3 Purity");
        gPur.attr().setTitleX(varName+" "+varUnits);
        gPur.attr().setTitleY(yTitle);

        GraphErrors gPur_l1 = new GraphErrors();
        gPur_l1.attr().setMarkerColor(8);
        gPur_l1.attr().setMarkerSize(10);
        gPur_l1.attr().setTitle("Level1 Purity");
        gPur_l1.attr().setTitleX(varName+" "+varUnits);
        gPur_l1.attr().setTitleY(yTitle);

        for (double q2=low;q2<high;q2+=step){
            INDArray metrics=Level3Tester.getMetsForBin(outputs,data.getLabels()[0],LabelVal,thresh, elClass,cutVar,q2,q2+step);
            gEff.addPoint(q2+step/2, metrics.getFloat(1, 0), 0, 0);
            gPur.addPoint(q2+step/2, metrics.getFloat(0, 0), 0, 0);
            gEff_l1.addPoint(q2+step/2, metrics.getFloat(3, 0), 0, 0);
            gPur_l1.addPoint(q2+step/2, metrics.getFloat(2, 0), 0, 0);
        } // Increment threshold on response

        TGCanvas c = new TGCanvas();
        c.setTitle("Efficiency vs "+varName);
        c.draw(gEff);
        if(addPur){c.draw(gPur, "same");}
        if(addL1){
            c.draw(gEff_l1, "same");
            if(addPur){c.draw(gPur_l1, "same");}
        }
        c.region().showLegend(0.05, 0.95);
        

    }


    public void test(MultiDataSet data,double thresh,int elClass,int LabelVal, String Part) {
        Boolean compL1=true;
        String varName="Q^2";
        String unitName="[GeV^2]";
        if(Part!="Electron"){
            compL1=false;
            varName="P";
            unitName="[GeV]";
        }
        
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        Level3Tester.PlotResponse(outputs[0], data.getLabels()[0],LabelVal,elClass,Part);
        Level3Tester.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,false,compL1,2,varName,unitName,0.0,11.0,1.);
        Level3Tester.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,compL1,3,"Sector","",0.5,6.5,1.0);
        INDArray metrics=Level3Tester.getMetrics(outputs[0],data.getLabels()[0],LabelVal,thresh, elClass);
        System.out.printf("\n Threshold: %f\n", thresh);
        System.out.printf("Level3 Purity: %f Efficiency: %f\n",metrics.getFloat(0,0),metrics.getFloat(1,0));
        System.out.printf("TP: %f, FP: %f, FN: %f\n",metrics.getFloat(4,0),metrics.getFloat(5,0),metrics.getFloat(6,0));
        System.out.printf("Level1 Purity: %f Efficiency: %f\n\n",metrics.getFloat(2,0),metrics.getFloat(3,0));
    }
    
    public static void main(String[] args){        
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018437_AI/rec_clas_018437.evio.00005-00009.hipo";//_AI
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018331_AI/rec_clas_018331.evio.00105-00109.hipo";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018326/run_018326_2.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018740/run_018740.h5";
        String file2="/Users/tyson/data_repo/trigger_data/rgd/018777/run_018777.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018432/run_018432.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/sims/el_rec.hipo";

        //String file2="/Users/tyson/data_repo/trigger_data/rga/rec_clas_005197.evio.00005-00009.hipo";


        Level3Tester t=new Level3Tester();
        //t.load("level3_0d_wrongTrigger.network");
        t.load("level3_0d_v3.network");//_v3 best
        //t.load("level3_0d_in.network");//_wrongTrigger
        //t.load("level3_0d_4C_t7_t4_t2t3_t1.network");
        //t.load("level3_0d_4C_t7_t4_t2t3_t1_in.network");

        //Get vals for electron
        int elClass=1;//1 for 2 classes, 2 for 3 classes, 3 for 4 classes
        int elLabelVal=1;
        MultiDataSet data=Level3Tester.getData(file2, 10000,10.547);
        double bestTh=t.findBestThreshold(data,elClass,0.995,elLabelVal);
        t.test(data, bestTh, elClass,elLabelVal,"Electron");//0.09

        //Get vals for other tracks
        /*int otherClass=2;//2 for 4 classes
        int otherLableVal=2;
        double bestThOther=t.findBestThreshold(data,otherClass,0.9,otherLableVal);
        t.test(data, bestThOther, otherClass,otherLableVal,"Charged Track");//0.09*/
        
    }
}
