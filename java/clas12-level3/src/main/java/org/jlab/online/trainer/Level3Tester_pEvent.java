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
public class Level3Tester_pEvent {

    

    public Level3Tester_pEvent() {

    }

    public static ComputationGraph load(String file) {
        ComputationGraph network=null;
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Tester_pEvent.class.getName()).log(Level.SEVERE, null, ex);
        }
        return network;
    }

    
   
    public static INDArray getLabels(String file,ComputationGraph network,int max,double beamE,double thresh,int elClass){

        HipoReader r = new HipoReader(file);
        Event e = new Event();

        int nMax = max;
        if (r.entries() < max)
            nMax = r.entries();

        INDArray LabelsArray = Nd4j.zeros(nMax, 8);
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter","REC::Cherenkov","ECAL::clusters");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        
        int counter=0,nEls=0,nL1=0,nNeg=0,nOther=0,nEmpty=0;
        while(r.hasNext() && counter<nMax){
            
            r.nextEvent(e);
            e.read(banks);
            e.read(dsts);

            long bits = banks[2].getLong("trigger", 0);
            int[] trigger = Level3Converter_MultiClass.convertTriggerLong(bits);
            //System.out.println(Arrays.toString(trigger));

            //list of helpful arrays
            List<Integer> elSectors = new ArrayList<Integer>();
            List<Integer> negSectors = new ArrayList<Integer>();
            List<Integer> otherSectors = new ArrayList<Integer>();
            Map<Integer, Level3Particle> fstElSector=new HashMap<Integer, Level3Particle>();
            Map<Integer, Level3Particle> fstOtherSector=new HashMap<Integer, Level3Particle>();
            Map<Integer, Level3Particle> fstNegSector=new HashMap<Integer, Level3Particle>();

            for(int i=0;i<dsts[0].getRows();i++){
                Level3Particle part=new Level3Particle();
                part.read_Particle_Bank(i,dsts[0]);
                part.read_Cal_Bank(dsts[2]);
                part.read_HTCC_bank(dsts[3]);
                part.find_sector_track(dsts[1]);
                if(part.PID==11){
                    //if(part.check_Energy_Dep_Cut()==true && part.check_FID_Cal_Clusters(dsts[4])==true){
                    //if(part.Nphe>2){
                        elSectors.add(part.Sector);
                        fstElSector.put(part.Sector,part);
                    //}
                } else if(part.PID!=2112 && part.PID!=22 && part.PID!=0 && part.PID!=-11){
                    otherSectors.add(part.Sector);
                    fstOtherSector.put(part.Sector,part);
                }
                if(part.Charge<0){
                    if(part.check_SF_cut()==true){
                        if(part.check_Energy_Dep_Cut()==true && part.check_FID_Cal_Clusters(dsts[4])==true){
                            negSectors.add(part.Sector);
                            fstNegSector.put(part.Sector,part);
                        }
                    }
                }
            }


            INDArray DCArray = Nd4j.zeros(6, 1, 6, 112);
            INDArray ECArray = Nd4j.zeros(6, 1, 6, 72);

            //loop over sectors
            for (int sect = 1; sect < 7; sect++) {

                // get DC and EC for the sector
                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                double nEdep = Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);

                
                Level3Utils.fillDC(DCArray, nodeDC, sect, sect-1);
                Level3Utils.fillEC(ECArray, nodeEC, sect, sect-1);

            }

            //if (trigger[30] == 1) {
                INDArray[] outputs = network.output(DCArray, ECArray);
                int gotEl = 0;
                double p = 0;
                double nphe=0;
                if (elSectors.size() > 0) {
                    gotEl = 1;
                    p = fstElSector.get(elSectors.get(0)).P;
                    nphe = fstElSector.get(elSectors.get(0)).Nphe;
                    nEls++;
                } else if (otherSectors.size() > 0) {
                    p = fstOtherSector.get(otherSectors.get(0)).P;
                    nphe = fstOtherSector.get(otherSectors.get(0)).Nphe;
                    gotEl = 2;
                    nOther++;
                } else{
                    nEmpty++;
                }

                int gotNeg = 0;
                double negp = 0;
                double negnphe=0;
                if (negSectors.size() > 0) {
                    gotNeg = 1;
                    negp = fstNegSector.get(negSectors.get(0)).P;
                    negnphe = fstNegSector.get(negSectors.get(0)).Nphe;
                    nNeg++;
                } else if (otherSectors.size() > 0) {
                    negp = fstOtherSector.get(otherSectors.get(0)).P;
                    negnphe = fstOtherSector.get(otherSectors.get(0)).Nphe;
                } 

                LabelsArray.putScalar(new int[] { counter, 0 }, gotEl);
                LabelsArray.putScalar(new int[] { counter, 1 }, trigger[7]);
                if (trigger[7] == 1) {
                    nL1++;
                }
                LabelsArray.putScalar(new int[] { counter, 2 }, p);
                LabelsArray.putScalar(new int[] { counter, 4 }, nphe);

                int l3Label = 0;
                for (int i = 0; i < 6; i++) {
                    if (outputs[0].getFloat(i, elClass) > thresh) {
                        l3Label = 1;
                    }
                }
                LabelsArray.putScalar(new int[] { counter, 3 }, l3Label);

                LabelsArray.putScalar(new int[] { counter, 5 }, gotNeg);
                LabelsArray.putScalar(new int[] { counter, 6 }, negp);
                LabelsArray.putScalar(new int[] { counter, 7 }, negnphe);
                counter++;
            //}


        }
        //System.out.print(OUTArray);
        //System.out.print(DCArray);
        //System.out.print(ECArray);
        System.out.printf("counter %d, nEl %d, nOther %d, nEmpty %d, nL1 %d, nEl_v2 %d\n\n",counter,nEls, nOther, nEmpty,nL1,nNeg);

        return LabelsArray;
    }

    public static void PlotResponse(INDArray Labels,int LabelVal,int LabelCol,String part) {
        long NEvents = Labels.shape()[0];
        H1F hRespPos = new H1F(part+" in Event", 101, 0, 1.01);
        hRespPos.attr().setLineColor(2);
        hRespPos.attr().setFillColor(2);
        hRespPos.attr().setLineWidth(3);
        hRespPos.attr().setTitleX("Response");
        H1F hRespNeg = new H1F("No "+part+" in Event", 101, 0, 1.01);
        hRespNeg.attr().setLineColor(5);
        hRespNeg.attr().setLineWidth(3);
        hRespNeg.attr().setTitleX("Response");
        //Sort predictions into those made on the positive/or negative samples
        for(long i=0;i<NEvents;i+=1) {
            if(Labels.getFloat(i,LabelCol)==LabelVal) {
            hRespPos.fill(Labels.getFloat(i,3));
            } else {
            hRespNeg.fill(Labels.getFloat(i,3));
            }
        }
    
        TGCanvas c = new TGCanvas();
        
        c.setTitle("Response");
        c.draw(hRespPos).draw(hRespNeg,"same");
        c.region().showLegend(0.05, 0.95);
            
        }//End of PlotResponse

    public static void PlotNphe(INDArray Labels,int NpheCol) {
        long NEvents = Labels.shape()[0];
        H1F hRespPos = new H1F("Selected by Trigger", 100, 0, 100);
        hRespPos.attr().setLineColor(2);
        hRespPos.attr().setFillColor(2);
        hRespPos.attr().setLineWidth(3);
        hRespPos.attr().setTitleX("Number of Photoelectrons");
        H1F hRespNeg = new H1F("Not Selected by Trigger", 100, 0, 100);
        hRespNeg.attr().setLineColor(5);
        hRespNeg.attr().setLineWidth(3);
        hRespNeg.attr().setTitleX("Number of Photoelectrons");
        //Sort predictions into those made on the positive/or negative samples
        for(long i=0;i<NEvents;i+=1) {
            if(Labels.getFloat(i,3)==1) {
            hRespPos.fill(Labels.getFloat(i,NpheCol));
            } else {
            hRespNeg.fill(Labels.getFloat(i,4));
            }
        }
    
        TGCanvas c = new TGCanvas();
        
        c.setTitle("Number of Photoelectrons");
        c.draw(hRespPos).draw(hRespNeg,"same");
        c.region().showLegend(0.05, 0.95);
            
        }//End of PlotResponse

    //Labels col 0 is 1 if there's an e-, 0 otherwise
    //Labels col 1 is 1 if l1 trigger fired, 0 otherwise
    //Labels col 2 is q2 when col 1 is 1, 0 otherwise
    public static INDArray getMetsForBin(INDArray Labels,int LabelVal,int LabelCol,int cutVar,double low,double high){
        INDArray metrics = Nd4j.zeros(4,1);
        long nEvents = Labels.shape()[0];

        double TP=0,FN=0,FP=0;
        double TP_l1=0,FN_l1=0,FP_l1=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, cutVar) > low && Labels.getFloat(i,cutVar)<high) {
                if (Labels.getFloat(i, LabelCol) == LabelVal) {
                    if (Labels.getFloat(i, 3) ==1) {
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
                    if (Labels.getFloat(i, 3) ==1) {
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


        //System.out.printf("Cut Val %f pur %f eff %f",low,metrics.getFloat(0, 0),metrics.getFloat(1, 0));
        //System.out.printf("TP %f FN %f FP %f\n", TP,FN,FP);

        return metrics;
    }

    //Labels col 0 is 1 if there's an e-, 0 otherwise
    //Labels col 1 is 1 if l1 trigger fired, 0 otherwise
    public static INDArray getMetrics(INDArray Labels,int LabelVal, int LabelCol){
        INDArray metrics = Nd4j.zeros(7,1);
        long nEvents = Labels.shape()[0];

        int nEls=0,nTrig=0;
        double TP=0,FP=0,FN=0;
        double TP_l1=0,FP_l1=0,FN_l1=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, LabelCol) == LabelVal) {
                nEls++;
                if (Labels.getFloat(i, 3) == 1) {
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
                if (Labels.getFloat(i, 3) == 1) {
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

    public static void plotVarDep(INDArray Labels,int LabelVal, int LabelCol,
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
            INDArray metrics=Level3Tester_pEvent.getMetsForBin(Labels,LabelVal,LabelCol,cutVar,q2,q2+step);
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
        c.region().showLegend(0.65, 0.25);
        

    }


    public void test(INDArray Labels,int LabelVal, int LabelCol, String Part) {
        /*Boolean compL1=true;
        String varName="Q^2";
        String unitName="[GeV^2]";
        if(Part!="Electron"){
            compL1=false;
            varName="P";
            unitName="[GeV]";
        }*/

        Boolean compL1=false;
        String varName="P";
        String unitName="[GeV]";

        int npheCol=4;
        int pCol=2;
        if(LabelCol==5){
            npheCol=7;
            pCol=6;
        }

        
        
        Level3Tester_pEvent.PlotResponse(Labels,LabelVal, LabelCol,Part);
        Level3Tester_pEvent.PlotNphe(Labels,npheCol);
        Level3Tester_pEvent.plotVarDep(Labels,LabelVal,LabelCol,true,compL1,pCol,varName,unitName,1.0,9.0,1.);
        INDArray metrics=Level3Tester_pEvent.getMetrics(Labels,LabelVal,LabelCol);

        
        System.out.printf("Level3 Purity: %f Efficiency: %f\n",metrics.getFloat(0,0),metrics.getFloat(1,0));
        System.out.printf("TP: %f, FP: %f, FN: %f\n",metrics.getFloat(4,0),metrics.getFloat(5,0),metrics.getFloat(6,0));
        System.out.printf("Level1 Purity: %f Efficiency: %f\n\n",metrics.getFloat(2,0),metrics.getFloat(3,0));
    }
    
    public static void main(String[] args){        
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018437_AI/rec_clas_018437.evio.00005-00009.hipo";//_AI
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018437/rec_clas_018437.evio.00005-00009.hipo";//_AI
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018331_AI/rec_clas_018331.evio.00105-00109.hipo";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018326/run_018326_2.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018740/run_018740.h5";
        String file2="/Users/tyson/data_repo/trigger_data/rgd/018777/run_018777.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018432/run_018432.h5";
        //String file2="/Users/tyson/data_repo/trigger_data/sims/el_rec.hipo";

        //String file2="/Users/tyson/data_repo/trigger_data/rga/rec_clas_005197.evio.00005-00009.hipo";

        

        Level3Tester_pEvent t=new Level3Tester_pEvent();
        //ComputationGraph network=t.load("level3_0d_wrongTrigger.network");
        ComputationGraph network=t.load("level3_0d_v3.network");//_v3 best
        //ComputationGraph network=t.load("level3_0d_in.network");//_wrongTrigger
        //ComputationGraph network=t.load("level3_0d_4C_t7_t4_t2t3_t1.network");
        //ComputationGraph network=t.load("level3_0d_4C_t7_t4_t2t3_t1_in.network");

        //Get vals for electron
        int elClass=1;//1 for 2 classes, 2 for 3 classes, 3 for 4 classes
        int elLabelVal=1;
        double thresh=0.6;//0.58

        
        INDArray Labels=Level3Tester_pEvent.getLabels(file2,network, 600000,10.547,thresh,elClass);

        System.out.printf("\n Event builder PID, Threshold: %f\n", thresh);
        t.test(Labels,elLabelVal,0,"Electron");//0.09

        System.out.printf("\n Own PID, Threshold: %f\n", thresh);
        t.test(Labels,elLabelVal,5,"Electron");
        
    }
}
