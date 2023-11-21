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
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu.add;
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
public class Level3Tester_Simulation {

    ComputationGraph network = null;

    public Level3Tester_Simulation() {

    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Tester.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    
   
    public static MultiDataSet getData(List<String[]> files,List<Integer[]> maxes,List<Integer> Classes, List<Integer> Sectors,double beamE,double trainTestP,Boolean mixMatchTracks){
        INDArray[] inputs = new INDArray[2];
        INDArray[] outputs = new INDArray[1];
        int nEls = 0, nOther = 0, added_files = 0, classs = 0,counter_tot=0,nBg=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d", classs);

            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j] + "_rec.hipo";

                HipoReader r = new HipoReader(file);
                Event e = new Event();

                int nMax = maxes.get(classs)[j];
                int start=(int)Math.ceil(trainTestP*r.entries());

                if (r.entries()< (nMax+start))
                    nMax = (r.entries()-start);

                INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray OUTArray = Nd4j.zeros(nMax, 5);

                Bank[] banks = r.getBanks("DC::tdc", "ECAL::adc", "RUN::config");
                Bank[] dsts = r.getBanks("REC::Particle", "REC::Track", "REC::Calorimeter", "REC::Cherenkov",
                        "ECAL::clusters","MC::Particle");

                CompositeNode nodeDC = new CompositeNode(12, 1, "bbsbil", 4096);
                CompositeNode nodeEC = new CompositeNode(11, 2, "bbsbifs", 4096);

                int counter = 0,eventNb=0;

                while (r.hasNext() && counter<nMax) {

                    r.nextEvent(e);
                    e.read(banks);

                    e.read(dsts);

                    if (eventNb >= start) {

                        List<Level3Particle> particles = new ArrayList<Level3Particle>();

                        // find and initialise particles
                        for (int i = 0; i < dsts[0].getRows(); i++) {
                            Level3Particle part = new Level3Particle();
                            part.read_Particle_Bank(i, dsts[0]);
                            part.read_MCParticle_Bank(0, dsts[5]);
                            part.read_Cal_Bank(dsts[2]);
                            part.read_HTCC_bank(dsts[3]);
                            part.find_sector_cal(dsts[2]);
                            particles.add(part);
                        }

                        // loop over sectors
                        for (int sect : Sectors) {

                            double p = 0;
                            double theta = 0;
                            double nphe=0;
                            Boolean keepEvent = false;
                            
                            // keep sectors with at least one particle
                            for (Level3Particle part : particles) {
                                if (part.Sector == sect) {
                                    
                                    if (part.TruthMatch(0.1, 0.1, 0.1)) {
                                        if (Classes.get(classs) == 1) {
                                            if (part.P > 0.5) {
                                                if (part.check_Energy_Dep_Cut() == true
                                                        && part.check_FID_Cal_Clusters(dsts[4]) == true
                                                        && part.check_SF_cut() == true) {
                                                    keepEvent = true;
                                                    p = part.P;
                                                    theta = part.Theta * (180.0 / Math.PI);
                                                    nphe=part.Nphe;
                                                    //System.out.printf("Nphe %d \f",nphe);
                                                }
                                            }
                                        } else {
                                            keepEvent = true;
                                            p = part.P;
                                            theta = part.Theta * (180.0 / Math.PI);
                                            nphe=part.Nphe;
                                        }
                                    }
                                }
                            }

                            Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                            Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                            // nodeDC.print();
                            // nodeEC.print();

                            if (nodeEC.getRows() > 0 && keepEvent == true) {

                                Level3Utils.fillDC(DCArray, nodeDC, sect, counter);
                                Level3Utils.fillEC(ECArray, nodeEC, sect, counter);

                                INDArray EventDCArray = DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),
                                        NDArrayIndex.all(), NDArrayIndex.all());
                                INDArray EventECArray = ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),
                                        NDArrayIndex.all(), NDArrayIndex.all());

                                // && hasL1==1
                                // check that the images aren't all empty, allow empty DC for neutrals
                                if (EventECArray.any()) { // EventDCArray.any()

                                    if (Classes.get(classs) == 1) {
                                        nEls++;
                                    }
                                    else if (Classes.get(classs) == 2) {
                                        nOther++;
                                    }
                                    else if (Classes.get(classs) == 0) {
                                        nBg++;
                                    }

                                    int nphe_mask=1;
                                    if(nphe<2.0){nphe_mask=0;}

                                    OUTArray.putScalar(new int[] { counter, 0 }, Classes.get(classs));
                                    OUTArray.putScalar(new int[] { counter, 1 }, p);
                                    OUTArray.putScalar(new int[] { counter, 2 }, theta);
                                    OUTArray.putScalar(new int[] { counter, 3 }, sect);
                                    OUTArray.putScalar(new int[] { counter, 4 }, nphe_mask);
                                    counter++;
                                    counter_tot++;
                                } else {
                                    // erase last entry
                                    DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 112));
                                    ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 72));
                                }
                            }
                        }
                    }
                    eventNb++;
                }
                System.out.printf("loaded samples (%d)\n\n\n", counter);
                if (added_files == 0) {
                    inputs = new INDArray[] { DCArray, ECArray };
                    outputs = new INDArray[] { OUTArray };
                } else {
                    inputs[0] = Nd4j.vstack(inputs[0], DCArray);
                    if(mixMatchTracks==true && Classes.get(classs)==1){
                        inputs[0] = Nd4j.vstack(DCArray, DCArray);
                    }
                    inputs[1] = Nd4j.vstack(inputs[1], ECArray);
                    outputs[0] = Nd4j.vstack(outputs[0], OUTArray);
                }
                added_files++;
            }
            classs++;
        }
        //System.out.print(OUTArray);
        //System.out.print(DCArray);
        //System.out.print(ECArray);
        System.out.printf("counter %d, nEl %d, nBg %d, nOther %d\n\n",counter_tot,nEls,nBg,nOther);

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
        INDArray metrics = Nd4j.zeros(2,1);
        long nEvents = outputs.shape()[0];

        double TP=0,FN=0,FP=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, cutVar) > low && Labels.getFloat(i,cutVar)<high) {
                if (Labels.getFloat(i, 0) == LabelVal) {
                    if (outputs.getFloat(i, elClass) > thresh) {
                        TP++;
                    } else {
                        FN++;
                    } // Check model prediction
                } else {
                    if (outputs.getFloat(i, elClass) > thresh) {
                        FP++;
                    }
                } // Check true label
            }
        }
	    double Pur=TP/(TP+FP);
	    double Eff=TP/(TP+FN);
	    metrics.putScalar(new int[] {0,0}, Pur);
	    metrics.putScalar(new int[] {1,0}, Eff);

        return metrics;
    }

    //Labels col 0 is 1 if there's an e-, 0 otherwise
    //Labels col 1 is 1 if l1 trigger fired, 0 otherwise
    public static INDArray getMetrics(INDArray outputs, INDArray Labels,int LabelVal,double thresh,int elClass){
        INDArray metrics = Nd4j.zeros(5,1);
        long nEvents = outputs.shape()[0];

        int nEls=0;
        double TP=0,FP=0,FN=0;
        for (int i = 0; i < nEvents; i++) {
            if (Labels.getFloat(i, 0) == LabelVal) {
                nEls++;
                if (outputs.getFloat(i, elClass) > thresh) {
                    TP++;
                } else {
                    FN++;
                } 
            } else {
                if (outputs.getFloat(i, elClass) > thresh) {
                    FP++;
                } 
            } // Check true label
        }
	    double Pur=TP/(TP+FP);
	    double Eff=TP/(TP+FN);
	    metrics.putScalar(new int[] {0,0}, Pur);
	    metrics.putScalar(new int[] {1,0}, Eff);
        metrics.putScalar(new int[] {2,0}, TP);
	    metrics.putScalar(new int[] {3,0}, FP);
        metrics.putScalar(new int[] {4,0}, FN);

        /*System.out.printf("Theres %d electrons in sample\n", nEls);
        System.out.printf("L1 trigger fired %d times in sample\n", nTrig);*/
        return metrics;
    }

    public double findBestThreshold(MultiDataSet data,int elClass,double effLow,int LabelVal,Boolean mask_nphe){

        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);

        if (mask_nphe) {
            INDArray mask=data.getLabels()[0].get(NDArrayIndex.all(), NDArrayIndex.point(4));
            mask=Nd4j.vstack(mask,mask);
            mask=mask.transpose();
            outputs[0] = outputs[0].mul(mask);
        }
        
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
        c.region().showLegend(0.25, 0.25);

        return bestRespTh;
    }

    public static void plotVarDep(MultiDataSet data, INDArray outputs, double thresh, int elClass, int LabelVal,
            Boolean addPur, int cutVar, String varName, String varUnits,double low, double high,double step) {

        String yTitle="Metrics";
        if(!addPur){yTitle="Efficiency";}

        GraphErrors gEff = new GraphErrors();
        gEff.attr().setMarkerColor(2);
        gEff.attr().setMarkerSize(10);
        gEff.attr().setTitle("Level3 Efficiency");
        gEff.attr().setTitleX(varName+" "+varUnits);
        gEff.attr().setTitleY(yTitle);

        GraphErrors gPur = new GraphErrors();
        gPur.attr().setMarkerColor(5);
        gPur.attr().setMarkerSize(10);
        gPur.attr().setTitle("Level3 Purity");
        gPur.attr().setTitleX(varName+" "+varUnits);
        gPur.attr().setTitleY(yTitle);


        for (double q2=low;q2<high;q2+=step){
            INDArray metrics=Level3Tester.getMetsForBin(outputs,data.getLabels()[0],LabelVal,thresh, elClass,cutVar,q2,q2+step);
            gEff.addPoint(q2+step/2, metrics.getFloat(1, 0), 0, 0);
            gPur.addPoint(q2+step/2, metrics.getFloat(0, 0), 0, 0);
        } // Increment threshold on response

        

        TGCanvas c = new TGCanvas();
        c.setTitle("Efficiency vs "+varName);
        c.draw(gEff);
        if(addPur){c.draw(gPur, "same");}
        c.region().axisLimitsY(gPur.getVectorY().getMin()-0.1, 1.05);
        c.region().showLegend(0.6, 0.25);
        

    }


    public void test(MultiDataSet data,double thresh,int elClass,int LabelVal, String Part,Boolean mask_nphe) {
        
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);

        if (mask_nphe) {
            INDArray mask=data.getLabels()[0].get(NDArrayIndex.all(), NDArrayIndex.point(4));
            mask=Nd4j.vstack(mask,mask);
            mask=mask.transpose();

            /*System.out.println(outputs[0]);
            System.out.println("\n\n");
            System.out.println(mask);
            System.out.println("\n\n");*/
            outputs[0] = outputs[0].mul(mask);
            //System.out.println(outputs[0]);
        }

        Level3Tester_Simulation.PlotResponse(outputs[0], data.getLabels()[0],LabelVal,elClass,Part);
        Level3Tester_Simulation.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,1,"P","[GeV]",1,9.0,1.0);
        Level3Tester_Simulation.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,2,"Theta","[Deg]",10.0,35.0,5.);
        //Level3Tester_Simulation.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,3,"Sector","",0.5,6.5,1.0);
        INDArray metrics=Level3Tester_Simulation.getMetrics(outputs[0],data.getLabels()[0],LabelVal,thresh, elClass);
        System.out.printf("\n Threshold: %f\n", thresh);
        System.out.printf("Level3 Purity: %f Efficiency: %f\n",metrics.getFloat(0,0),metrics.getFloat(1,0));
        System.out.printf("TP: %f, FP: %f, FN: %f\n",metrics.getFloat(2,0),metrics.getFloat(3,0),metrics.getFloat(4,0));
    }
    
    public static void main(String[] args){        
        
        String dir = "/Users/tyson/data_repo/trigger_data/sims/";
        String out = "/Users/tyson/data_repo/trigger_data/sims/python/";

        List<String[]> files = new ArrayList<>();
        files.add(new String[] {dir+"pim",dir+"gamma",dir+"pos" });//dir+"pim"
        files.add(new String[] { dir+"el" });

        /*files.add(new String[] { dir+"gamma"});
        files.add(new String[] { dir+"el" });*/

        List<Integer[]> maxes = new ArrayList<>();
        maxes.add(new Integer[] {1600,1600,1600});
        maxes.add(new Integer[] {4800});

        /*maxes.add(new Integer[] {4800});
        maxes.add(new Integer[] {4800});*/

        List<Integer> classes=new ArrayList<>();
        classes.add(0);
        classes.add(1);

        List<Integer> sectors=new ArrayList<Integer>(); //simulated only in sectors 1 and 6
        sectors.add(1);

        //String file2="/Users/tyson/data_repo/trigger_data/rga/rec_clas_005197.evio.00005-00009.hipo";

        Level3Tester_Simulation t=new Level3Tester_Simulation();
        //t.load("level3_sim_0d.network");
        t.load("level3_0d_in.network");

        Boolean mask_nphe=false;
        //NB: this only works with two samples of same size and assuming the electron sample comes last
        Boolean mixMatchTracks=false; 

        //Get vals for electron
        int elClass=1;//1 for 2 classes, 2 for 3 classes, 3 for 4 classes
        int elLabelVal=1;
        MultiDataSet data=Level3Tester_Simulation.getData(files,maxes,classes,sectors,10.547,0.8,mixMatchTracks);
        double bestTh=t.findBestThreshold(data,elClass,0.995,elLabelVal,mask_nphe);
        t.test(data, bestTh, elClass,elLabelVal,"Electron",mask_nphe);//bestTh

        //Get vals for other tracks
        /*int otherClass=2;//2 for 4 classes
        int otherLableVal=2;
        double bestThOther=t.findBestThreshold(data,otherClass,0.9,otherLableVal);
        t.test(data, bestThOther, otherClass,otherLableVal,"Charged Track");//0.09*/
        
    }
}
