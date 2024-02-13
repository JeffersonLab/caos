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
import twig.data.H2F;
import twig.graphics.TGCanvas;

/**
 *
 * @author tyson
 */
public class Level3Tester_SimulationSIDIS_MCPart {

    ComputationGraph network = null;

    public Level3Tester_SimulationSIDIS_MCPart() {

    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Tester.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static int nPart_pSect(List<Level3Particle> particles,int sect){
        int nPart_pSect=0;
        for(Level3Particle part:particles){
            if(part.Cal_Sector==sect){
                nPart_pSect++;
            }
        }
        return nPart_pSect;

    }

    public static int nMCPart_pSect(List<Level3Particle> particles,int sect){
        int nPart_pSect=0;
        for(Level3Particle part:particles){
            if(part.MC_Sector==sect){
                nPart_pSect++;
            }
        }
        return nPart_pSect;

    }

    public static int nTrack_pSect(Bank TrackBank,int sect){
        int nTrack_pSect=0;
        for (int k = 0; k < TrackBank.getRows(); k++) {
            int pindex = TrackBank.getInt("pindex", k);
            int sectorTrk = TrackBank.getInt("sector", k);
            if(sectorTrk==sect){nTrack_pSect++;}
        }
        return nTrack_pSect;

    }

    public static List<Level3Particle> getMCPart_pSect(Bank[] dsts,int sect){
        List<Level3Particle> pInS = new ArrayList<Level3Particle>();
        // find and initialise particles
        for (int i = 0; i < dsts[5].getRows(); i++) {
            Level3Particle part = new Level3Particle();
            part.read_MCParticle_Bank(i, dsts[5]);
            part.find_ClosestRECParticle(dsts[0]);
            if (part.PIndex != -1) { // ie REC part in FD
                part.read_Cal_Bank(dsts[2]);
                part.read_HTCC_bank(dsts[3]);
                part.find_sector_cal(dsts[2]);
                part.find_sector_track(dsts[1]);
            }
            // ie MC part in FD
            if(part.MC_Sector==sect) {
                pInS.add(part);
            }
        }
        return pInS;
    }

    public static List<Level3Particle> getRECPart_pSect(Bank[] dsts,int sect){
        List<Level3Particle> pInS = new ArrayList<Level3Particle>();

        // find and initialise particles
        for (int i = 0; i < dsts[0].getRows(); i++) {
            Level3Particle part = new Level3Particle();
            part.read_Particle_Bank(i, dsts[0]);
            if(part.PIndex!=-1){ //ie part in FD
                part.find_ClosestMCParticle(dsts[5]);
                part.read_Cal_Bank(dsts[2]);
                part.read_HTCC_bank(dsts[3]);
                part.find_sector_cal(dsts[2]);
                part.find_sector_track(dsts[1]);
                if(sect==part.Cal_Sector){
                    pInS.add(part);
                }
                
            }
        }
        return pInS;
    }

    public static Boolean noCloseMCPart(Level3Particle part, Bank MCBank){
        Boolean closeMCPart=false;
        for (int i = 0; i < MCBank.getRows(); i++) {
            if(i!=part.MC_PIndex){
                double MC_PID = MCBank.getInt("pid", i);
                double MC_Px = MCBank.getFloat("px", i);
                double MC_Py = MCBank.getFloat("py", i);
                double MC_Pz = MCBank.getFloat("pz", i);
                double MC_P = Math.sqrt(MC_Px * MC_Px + MC_Py * MC_Py + MC_Pz * MC_Pz);
                double MC_Theta = Math.acos(MC_Pz / MC_P);// Math.atan2(Math.sqrt(px*px+py*py),pz);
                double MC_Phi = Math.atan2(MC_Py, MC_Px);
                if(Math.abs(MC_Phi-part.MC_Phi)<1.05){ //0.5 rad ~30 deg, 1.05 rad ~ 60 deg
                    closeMCPart=true;
                }

            }
        }
        return closeMCPart;
    }

    public static MultiDataSet getData(String dir,int max,double beamE){
        int nEls = 0, nOther = 0,counter_tot=0,nBg=0;

        int start=1;

        // INDArray DCArray = Nd4j.zeros(max, 1, 6, 112);
        INDArray DCArray = Nd4j.zeros(max, 6, 6, 112);
        INDArray ECArray = Nd4j.zeros(max, 1, 6, 72);
        INDArray FTOFArray = Nd4j.zeros(max, 1, 62, 1);
        INDArray HTCCArray = Nd4j.zeros(max, 1, 8, 1);
        INDArray OUTArray = Nd4j.zeros(max, 10);

        while (counter_tot < max && start < 5) {//56

            String file = dir + "clasdis_" + String.valueOf(start) + ".hipo";
            //String file=dir;
            start++;

            HipoReader r = new HipoReader(file);
            Event e = new Event();

            // r.getSchemaFactory().show();

            CompositeNode nodeDC = new CompositeNode(12, 1, "bbsbil", 4096);
            CompositeNode nodeEC = new CompositeNode(11, 2, "bbsbifs", 4096);
            CompositeNode nodeFTOF = new CompositeNode(13, 3, "bbsbifs", 4096);
            CompositeNode nodeHTCC = new CompositeNode(14, 5, "bbsbifs", 4096);

            Bank[] banks = r.getBanks("DC::tdc", "ECAL::adc", "RUN::config", "FTOF::adc", "HTCC::adc");
            Bank[] dsts = r.getBanks("REC::Particle", "REC::Track", "REC::Calorimeter", "REC::Cherenkov",
                    "ECAL::clusters", "MC::Particle");

            while (r.hasNext() && counter_tot < max) {

                r.nextEvent(e);
                e.read(banks);

                //System.out.println("DC");
                //banks[0].show();

                /*
                 * System.out.println("FTOF");
                 * banks[3].show();
                 * System.out.println("HTCC");
                 * banks[4].show();
                 */

                e.read(dsts);

                // loop over sectors
                for (int sect = 1; sect < 7; sect++) {

                    double p = 0;
                    double theta = 0;
                    double phi = 0;
                    double nphe = 0;
                    int PID=0;
                    Boolean hasParticle = false,hasEl = false;
                    int classs=0;
                    double SF=0;
                    double ECAL_e=0;

                    List<Level3Particle> pInSREC=getRECPart_pSect(dsts, sect);
                    List<Level3Particle> pInSMC=getMCPart_pSect(dsts, sect);

                    // check if we have at least one REC particle
                    if (pInSREC.size() != 0) {
                        hasParticle = true;
                        //loop over MC particles
                        for (Level3Particle part : pInSMC) {
                            if (part.MC_PID == 11) {
                                p = part.MC_P;
                                theta = part.MC_Theta * (180.0 / Math.PI);
                                phi = part.MC_Phi * (180.0 / Math.PI);
                                PID = part.MC_PID;
                                //for electrons, check we have at least one REC particle with HTCC in same sector
                                for (Level3Particle recpart : pInSREC) {
                                    part.read_Particle_Bank(recpart.PIndex, dsts[0]);
                                    part.read_HTCC_bank(dsts[3]);
                                    part.find_sector_cal(dsts[2]);
                                    if(part.Nphe>2 && part.HTCC_Sector==part.Cal_Sector){
                                        SF = part.SF;
                                        ECAL_e = part.ECAL_energy;
                                        nphe = part.Nphe;
                                        hasEl = true;
                                    }
                                }
                                
                            } 
                            else {
                                p = part.MC_P;
                                theta = part.MC_Theta * (180.0 / Math.PI);
                                phi = part.MC_Phi * (180.0 / Math.PI);
                                PID = part.MC_PID;
                                // only take REC quantities for truthmatched particle
                                if (part.PIndex != -1 && part.TruthMatch(0.1, 0.1, 0.1)) {
                                    SF = part.SF;
                                    ECAL_e = part.ECAL_energy;
                                    nphe = part.Nphe;
                                }
                            }
                        }
                    }

                    // if an event has an electron
                    // don't care if it has other particles
                    // we want it to be in electron sample
                    if (hasParticle) {
                        if (hasEl) {
                            classs=1;
                        } else {
                            classs=2;
                        }
                    }

                    Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                    Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                    Level3Converter_MultiClass.convertFTOF(banks[3], nodeFTOF, sect);
                    Level3Converter_MultiClass.convertHTCC(banks[4], nodeHTCC, sect);
                    // nodeDC.print();
                    // nodeEC.print();
                    //nodeDC.getRows() > 0 &&
                    if ( nodeEC.getRows() > 0 && counter_tot<max  ) {//&& hasParticle == true && classs == 1 && classs == 1

                        // Level3Utils.fillDC_wLayers(DCArray, nodeDC, sect, counter_tot);
                        // Level3Utils.fillDC(DCArray, nodeDC, sect, counter_tot);
                        Level3Utils.fillDC_SepSL(DCArray, nodeDC, sect, counter_tot);
                        Level3Utils.fillEC(ECArray, nodeEC, sect, counter_tot);
                        Level3Utils.fillFTOF(FTOFArray, nodeFTOF, sect, counter_tot);
                        Level3Utils.fillHTCC(HTCCArray, nodeHTCC, sect, counter_tot);

                        INDArray EventDCArray = DCArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(),
                               NDArrayIndex.all(), NDArrayIndex.all());
                        INDArray EventECArray = ECArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(),
                               NDArrayIndex.all(), NDArrayIndex.all());

                        //System.out.println("filled arrays");
                        // check that DC & EC aren't all empty, should at least have noise
                        if (EventDCArray.any() && EventECArray.any()) { 

                            INDArray EventHTCCCArray = HTCCArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(),
                               NDArrayIndex.all(), NDArrayIndex.all());

                            if(classs==1 && !EventHTCCCArray.any()){
                                classs=0;
                            }

                            if (classs == 1) {
                                nEls++;
                            } else if (classs== 2) {
                                nOther++;
                            } else if (classs== 0) {
                                nBg++;
                            }

                            int nphe_mask = 1;
                            if (nphe < 2.0) {
                                nphe_mask = 0;
                            }

                            /*if((counter_tot%1000)==0){
                                System.out.printf("\nAdded %d events\n",counter_tot);
                            }*/
                            
                            OUTArray.putScalar(new int[] { counter_tot, 0 }, classs);
                            OUTArray.putScalar(new int[] { counter_tot, 1 }, p);
                            OUTArray.putScalar(new int[] { counter_tot, 2 }, theta);
                            OUTArray.putScalar(new int[] { counter_tot, 3 }, sect);
                            OUTArray.putScalar(new int[] { counter_tot, 4 }, nphe_mask);
                            OUTArray.putScalar(new int[] { counter_tot, 5 }, phi);
                            OUTArray.putScalar(new int[] { counter_tot, 6 }, nphe);
                            OUTArray.putScalar(new int[] { counter_tot, 7 }, PID);
                            OUTArray.putScalar(new int[] { counter_tot, 8 }, SF);
                            OUTArray.putScalar(new int[] { counter_tot, 9 }, ECAL_e);
                            counter_tot++;
                         } else {
                            // erase last entry
                            // DCArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(),
                            // NDArrayIndex.all(),
                            // NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 112));
                            DCArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Nd4j.zeros(6, 6, 112));
                            ECArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 72));
                            FTOFArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Nd4j.zeros(62));
                            HTCCArray.get(NDArrayIndex.point(counter_tot), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Nd4j.zeros(8));
                        }
                    }
                }
            }
        }
        // System.out.print(OUTArray);
        // System.out.print(DCArray);
        // System.out.print(ECArray);
        System.out.printf("counter %d, n events with e- %d, n empty events %d, n events with other particle type %d\n\n", counter_tot, nEls, nBg,
                nOther);

        MultiDataSet dataset = new MultiDataSet(new INDArray[] { DCArray, ECArray, FTOFArray, HTCCArray }, new INDArray[] { OUTArray });

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

        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        //0d_FTOFHTCC
        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[2], data.getFeatures()[3]);
        //0f
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[3]);

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
        //c.region().axisLimitsY(0.8, 1.01);

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

    public static void plotDCExamples(INDArray DCall, int nExamples,int start){
        for (int k = start; k < nExamples+start; k++) {
            for (int l = 0; l < DCall.shape()[1]; l++) {
                TGCanvas c = new TGCanvas();
                c.setTitle("DC (Superlayer "+String.valueOf(l)+")");

                H2F hDC = new H2F("DC", (int) DCall.shape()[3], 0, (int) DCall.shape()[3], (int) DCall.shape()[2], 0,
                        (int) DCall.shape()[2]);
                hDC.attr().setTitleX("Wires");
                hDC.attr().setTitleY("Layers");
                hDC.attr().setTitle("DC");
                INDArray DCArray = DCall.get(NDArrayIndex.point(k), NDArrayIndex.point(l),
                        NDArrayIndex.all(),
                        NDArrayIndex.all());
                for (int i = 0; i < DCArray.shape()[0]; i++) {
                    for (int j = 0; j < DCArray.shape()[1]; j++) {
                        if (DCArray.getFloat(i, j) != 0) {
                            hDC.fill(j, i, DCArray.getFloat(i, j));
                        }
                    }
                }
                c.draw(hDC);
            }
        }

	}

    public static void plotECExamples(INDArray ECall, int nExamples,int start){
        for (int k = start; k < nExamples+start; k++) {
            TGCanvas c = new TGCanvas();
            c.setTitle("ECAL");

            H2F hEC = new H2F("EC", (int) ECall.shape()[3], 0, (int) ECall.shape()[3], (int) ECall.shape()[2], 0,
                    (int) ECall.shape()[2]);
            hEC.attr().setTitleX("Strips");
            hEC.attr().setTitleY("Layers");
            hEC.attr().setTitle("EC");
            for (int l = 0; l < ECall.shape()[1]; l++) {
                for (int i = 0; i < ECall.shape()[2]; i++) {
                    for (int j = 0; j < ECall.shape()[3]; j++) {
                        if (ECall.getFloat(k,l,i, j) != 0) {
                            hEC.fill(j, i, ECall.getFloat(k,l,i, j));
                        }
                    }
                }
            }
            c.draw(hEC);
        }

	}


    public void test(MultiDataSet data,double thresh,int elClass,int LabelVal, String Part,Boolean mask_nphe) {
        
        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        //0d_FTOFHTCC
        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1],data.getFeatures()[2],data.getFeatures()[3]);
        //0f
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[3]);

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

        Level3Tester_SimulationSIDIS_MCPart.PlotResponse(outputs[0], data.getLabels()[0],LabelVal,elClass,Part);
        Level3Tester_SimulationSIDIS_MCPart.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,1,"P","[GeV]",1,9.0,1.0);
        Level3Tester_SimulationSIDIS_MCPart.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,2,"Theta","[Deg]",10.0,35.0,5.);
        //Level3Tester_SimulationSIDIS_MCPart.plotVarDep(data,outputs[0],thresh,elClass,LabelVal,true,3,"Sector","",0.5,6.5,1.0);
        INDArray metrics=Level3Tester_SimulationSIDIS_MCPart.getMetrics(outputs[0],data.getLabels()[0],LabelVal,thresh, elClass);
        System.out.printf("\n Threshold: %f\n", thresh);
        System.out.printf("Level3 Purity: %f Efficiency: %f\n",metrics.getFloat(0,0),metrics.getFloat(1,0));
        System.out.printf("TP: %f, FP: %f, FN: %f\n",metrics.getFloat(2,0),metrics.getFloat(3,0),metrics.getFloat(4,0));

        //debugElectrons(outputs[0], data,LabelVal,elClass);
        debugBG(outputs[0], data,LabelVal,elClass);
    }

    public static void plotElHTCCExamples(MultiDataSet data,int nEx){
        int nPlotted=0;
        for (int i=0;i<data.getLabels()[0].shape()[0];i++){
            if(data.getLabels()[0].getFloat(i, 0) == 1){
                if(nPlotted<nEx){
                    plotECExamples(data.getFeatures()[3], 1, i);
                    nPlotted++;
                }
            }
        }
    }

    public static void debugElectrons(INDArray outputs, MultiDataSet data,int elLabelVal,int elClass){
        int shown=0;
        INDArray Labels=data.getLabels()[0];
        for(int i=0;i<outputs.shape()[0];i++){
            if(Labels.getFloat(i, 0) == elLabelVal){
                if(outputs.getFloat(i, elClass) <0.001 && shown<1){
                    System.out.printf("\nElectron @ resp %f\n",outputs.getFloat(i, elClass));
                    System.out.printf("P %f, theta %f, phi %f\n",Labels.getFloat(i, 1),Labels.getFloat(i, 2),Labels.getFloat(i, 5));
                    System.out.printf("nphe %f, sect %f, PID %f\n",Labels.getFloat(i, 6),Labels.getFloat(i, 3),Labels.getFloat(i, 7));
                    System.out.printf("SF %f, ECAL energy %f\n",Labels.getFloat(i, 8),Labels.getFloat(i, 9));
                    plotDCExamples(data.getFeatures()[0], 1, i);
                    //plotECExamples(data.getFeatures()[1], 1, i);
                    //plotECExamples(data.getFeatures()[3], 1, i);
                    shown++;

                }
            }
        }

        int shown2=0;
        for(int i=0;i<outputs.shape()[0];i++){
            if(Labels.getFloat(i, 0) == elLabelVal){
                if(outputs.getFloat(i, elClass) >0.9 && shown2<1){
                    System.out.printf("\nElectron @ resp %f\n",outputs.getFloat(i, elClass));
                    System.out.printf("P %f, theta %f, phi %f\n",Labels.getFloat(i, 1),Labels.getFloat(i, 2),Labels.getFloat(i, 5));
                    System.out.printf("nphe %f, sect %f, PID %f\n",Labels.getFloat(i, 6),Labels.getFloat(i, 3),Labels.getFloat(i, 7));
                    System.out.printf("SF %f, ECAL energy %f\n",Labels.getFloat(i, 8),Labels.getFloat(i, 9));
                    plotDCExamples(data.getFeatures()[0], 1, i);
                    //plotECExamples(data.getFeatures()[1], 1, i);
                    //plotECExamples(data.getFeatures()[3], 1, i);
                    shown2++;

                }
            }
        }
    }

    public static void debugBG(INDArray outputs, MultiDataSet data,int elLabelVal,int elClass){

        H1F hBGType = new H1F("BG PID IDed as signal", 8, -0.5, 7.5);
        hBGType.attr().setLineColor(2);
        hBGType.attr().setFillColor(2);
        hBGType.attr().setLineWidth(3);
        hBGType.attr().setTitleX("BG PID (0 empty, 1 -211, 2 -11, 3 22, 4 11, 5 2112, 6 +/- 321, 7 others)");


        H1F hHasHTCC = new H1F("BG has HTCC?", 2, -0.5, 1.5);
        hHasHTCC.attr().setLineColor(2);
        hHasHTCC.attr().setFillColor(2);
        hHasHTCC.attr().setLineWidth(3);
        hHasHTCC.attr().setTitleX("HTCC not empty (0 no, 1 true)");
        hHasHTCC.attr().setTitle("Empty");

        H1F hothHasHTCC = new H1F("BG has HTCC?", 2, -0.5, 1.5);
        hothHasHTCC.attr().setLineColor(3);
        hothHasHTCC.attr().setLineWidth(3);
        hothHasHTCC.attr().setTitleX("HTCC not empty (0 no, 1 true)");
        hothHasHTCC.attr().setTitle("PID Other");

        H1F hphHasHTCC = new H1F("BG has HTCC?", 2, -0.5, 1.5);
        hphHasHTCC.attr().setLineColor(5);
        hphHasHTCC.attr().setLineWidth(3);
        hphHasHTCC.attr().setTitleX("HTCC not empty (0 no, 1 true)");
        hphHasHTCC.attr().setTitle("PID 22");

        H1F hnHasHTCC = new H1F("BG has HTCC?", 2, -0.5, 1.5);
        hnHasHTCC.attr().setLineColor(1);
        hnHasHTCC.attr().setLineWidth(3);
        hnHasHTCC.attr().setTitleX("HTCC not empty (0 no, 1 true)");
        hnHasHTCC.attr().setTitle("PID 2112");

        H1F hkHasHTCC = new H1F("BG has HTCC?", 2, -0.5, 1.5);
        hkHasHTCC.attr().setLineColor(6);
        hkHasHTCC.attr().setLineWidth(3);
        hkHasHTCC.attr().setTitleX("HTCC not empty (0 no, 1 true)");
        hkHasHTCC.attr().setTitle("PID +/- 321");
    
        

        int shown=0;
        INDArray Labels=data.getLabels()[0];
        for(int i=0;i<outputs.shape()[0];i++){
            if(Labels.getFloat(i, 0) != elLabelVal){
                if(outputs.getFloat(i, elClass) >0.9){

                    INDArray htcc=data.getFeatures()[3].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all());

                    int bgtype=-1;
                    if(Labels.getFloat(i, 7)==0){
                        bgtype=0;
                        
                        if(htcc.any()){
                            hHasHTCC.fill(1);
                        } else{
                            hHasHTCC.fill(0);
                        }

                    } else if(Labels.getFloat(i, 7)==-211){
                        bgtype=1;
                    }
                    else if(Labels.getFloat(i, 7)==-11){
                        bgtype=2;
                    }
                    else if(Labels.getFloat(i, 7)==22){
                        bgtype=3;

                        if(htcc.any()){
                            hphHasHTCC.fill(1);
                        } else{
                            hphHasHTCC.fill(0);
                        }

                    }else if(Labels.getFloat(i, 7)==11){
                        bgtype=4;
                    }
                    else if(Labels.getFloat(i, 7)==2112){
                        bgtype=5;

                        if(htcc.any()){
                            hnHasHTCC.fill(1);
                        } else{
                            hnHasHTCC.fill(0);
                        }

                    }
                    else if(Math.abs(Labels.getFloat(i, 7))==321){
                        bgtype=6;

                        if(htcc.any()){
                            hkHasHTCC.fill(1);
                        } else{
                            hkHasHTCC.fill(0);
                        }

                    }
                    else {
                        bgtype=7;

                        if(htcc.any()){
                            hothHasHTCC.fill(1);
                        } else{
                            hothHasHTCC.fill(0);
                        }

                    }

                    if(bgtype!=-1){
                        hBGType.fill(bgtype);
                    }

                
                    if(shown<1){
                        System.out.printf("\nBG @ resp %f\n",outputs.getFloat(i, elClass));
                        System.out.printf("P %f, theta %f, phi %f\n",Labels.getFloat(i, 1),Labels.getFloat(i, 2),Labels.getFloat(i, 5));
                        System.out.printf("nphe %f, sect %f, PID %f\n",Labels.getFloat(i, 6),Labels.getFloat(i, 3),Labels.getFloat(i, 7));
                        System.out.printf("SF %f, ECAL energy %f\n",Labels.getFloat(i, 8),Labels.getFloat(i, 9));
                        plotDCExamples(data.getFeatures()[0], 1, i);
                        plotECExamples(data.getFeatures()[1], 1, i);
                        plotECExamples(data.getFeatures()[3], 1, i);
                        shown++;
                    }

                }
            }
        }

        TGCanvas c = new TGCanvas();
        c.setTitle("BG Type");
        c.draw(hBGType);

        TGCanvas c2 = new TGCanvas();
        c2.setTitle("has HTCC");
        c2.draw(hHasHTCC).draw(hphHasHTCC,"same").draw(hothHasHTCC,"same").draw(hnHasHTCC,"same").draw(hkHasHTCC,"same");
        c2.region().showLegend(0.25, 0.6);

    }
    
    public static void main(String[] args){        
        
        String dir = "/Users/tyson/data_repo/trigger_data/sims/claspyth/";
        //String dir = "/Users/tyson/data_repo/trigger_data/sims/claspyth_train/";
        //String dir = "/Users/tyson/data_repo/trigger_data/rgd/018777/run_018777.h5";
        //String dir = "/Users/tyson/data_repo/trigger_data/rgd/018331_AI/rec_clas_018331.evio.00100-00104.hipo";

        Level3Tester_SimulationSIDIS_MCPart t=new Level3Tester_SimulationSIDIS_MCPart();
        //t.load("level3_sim_0d.network");
        //t.load("level3_sim_fullLayers_0d.network");
        //t.load("level3_0d_in.network");
        //t.load("level3_sim_0d_FTOFHTCC.network");
        //t.load("level3_sim_wMixMatch_0d_FTOFHTCC.network");
        //t.load("level3_sim_MC_wMixMatch_0d_FTOFHTCC_v1.network");
        t.load("level3_sim_MC_wMixMatch_wCorrupt_wbg_SIDIS_0f.network");//5C
        //t.load("level3_sim_MC_wMixMatch_wCorrupt_wbg_wEmpty_SIDIS_0f.network");//6C
        //t.load("level3_sim_MC_wCorrupt_wbg_wEmpty_SIDIS_0f.network");//5C

        Boolean mask_nphe=false;

        //Get vals for electron
        int elClass=4;//1 for 2 classes, 2 for 3 classes, 3 for 4 classes etc
        int elLabelVal=1;
        MultiDataSet data=Level3Tester_SimulationSIDIS_MCPart.getData(dir,10000,10.6);//10000

        //plotDCExamples(data.getFeatures()[0], 1,500);
        //plotECExamples(data.getFeatures()[1], 1,500);
        //plotElHTCCExamples(data,5);

        double bestTh=t.findBestThreshold(data,elClass,0.995,elLabelVal,mask_nphe);//0.995
        t.test(data, bestTh, elClass,elLabelVal,"Electron",mask_nphe);//bestTh

        //Get vals for other tracks
        /*int otherClass=0;//2 for 4 classes
        int otherLableVal=2;
        double bestThOther=t.findBestThreshold(data,otherClass,0.9,otherLableVal);
        t.test(data, bestThOther, otherClass,otherLableVal,"pi-");//0.09*/
        
    }
}
