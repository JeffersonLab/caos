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
public class Level3Tester_SingleEvent_SimulationSIDIS {

    ComputationGraph network = null;

    public Level3Tester_SingleEvent_SimulationSIDIS() {

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

    public static int nTrack_pSect(Bank TrackBank,int sect){
        int nTrack_pSect=0;
        for (int k = 0; k < TrackBank.getRows(); k++) {
            int pindex = TrackBank.getInt("pindex", k);
            int sectorTrk = TrackBank.getInt("sector", k);
            if(sectorTrk==sect){nTrack_pSect++;}
        }
        return nTrack_pSect;

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

    public void test(String file,double beamE, int evNb, int pindextofind,int elClass,int elLabelVal){
        int counter_tot=0,nbread=1;

        // INDArray DCArray = Nd4j.zeros(1, 1, 6, 112);
        INDArray DCArray = Nd4j.zeros(1, 6, 6, 112);
        INDArray ECArray = Nd4j.zeros(1, 1, 6, 72);
        INDArray FTOFArray = Nd4j.zeros(1, 1, 62, 1);
        INDArray HTCCArray = Nd4j.zeros(1, 1, 8, 1);
        INDArray OUTArray = Nd4j.zeros(1, 10);

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

        while (r.hasNext() && counter_tot < 1) {

            r.nextEvent(e);
            e.read(banks);

            // System.out.println("DC");
            // banks[0].show();

            /*
             * System.out.println("FTOF");
             * banks[3].show();
             * System.out.println("HTCC");
             * banks[4].show();
             */

            e.read(dsts);

            

            if (nbread == evNb) {

                System.out.println("Particle");
                dsts[0].show();

                System.out.println("MC");
                dsts[5].show();

                List<Level3Particle> particles = new ArrayList<Level3Particle>();

                double SF = 0;
                double ECAL_e = 0;
                double p = 0;
                double theta = 0;
                double phi = 0;
                double nphe = 0;
                int PID = 0;
                int MC_PID = 0;

                int sect=-1;
                // find and initialise particles
                for (int i = 0; i < dsts[0].getRows(); i++) {
                    Level3Particle part = new Level3Particle();
                    part.read_Particle_Bank(i, dsts[0]);
                    if (part.PIndex != -1) { // ie part in FD
                        part.find_ClosestMCParticle(dsts[5]);
                        part.read_Cal_Bank(dsts[2]);
                        part.read_HTCC_bank(dsts[3]);
                        part.find_sector_cal(dsts[2]);
                        part.find_sector_track(dsts[1]);
                        particles.add(part);
                    }
                    if (i==pindextofind){//part.PID
                        sect = part.Cal_Sector;
                        p = part.P;
                        theta = part.Theta * (180.0 / Math.PI);
                        phi = part.Phi * (180.0 / Math.PI);
                        nphe = part.Nphe;
                        MC_PID = part.MC_PID;
                        PID = part.PID;
                        SF = part.SF;
                        ECAL_e = part.ECAL_energy;
                    }
                }

                
                Boolean hasParticle = false, hasEl = false;
                int classs = 0;

                // check if we have at least one particle
                // and if there's an electron in the sector
                for (Level3Particle part : particles) {
                    if (part.Cal_Sector == sect) {// && nPart_pSect(particles, sect)==1) { //&&
                                                  // nTrack_pSect(dsts[1],sect)==1
                        if (part.TruthMatch(0.1, 0.1, 0.1)) {
                            if (part.P > 0.5) {
                                if (part.MC_PID == 11) {
                                    if (part.check_Energy_Dep_Cut() == true
                                            && part.check_FID_Cal_Clusters(dsts[4]) == true
                                            && part.check_SF_cut() == true
                                            && part.Nphe >= 2) {
                                        hasEl = true;
                                        hasParticle = true;
                                    }
                                } else {
                                    hasParticle = true;
                                }
                            }
                        }
                    }
                }

                // if an event has an electron
                // don't care if it has other particles
                // we want it to be in electron sample
                if (hasParticle) {
                    if (hasEl) {
                        classs = 1;
                    } else {
                        classs = 2;
                    }
                }

                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                Level3Converter_MultiClass.convertFTOF(banks[3], nodeFTOF, sect);
                Level3Converter_MultiClass.convertHTCC(banks[4], nodeHTCC, sect);

                // Level3Utils.fillDC_wLayers(DCArray, nodeDC, sect, counter_tot);
                // Level3Utils.fillDC(DCArray, nodeDC, sect, counter_tot);
                Level3Utils.fillDC_SepSL(DCArray, nodeDC, sect, counter_tot);
                Level3Utils.fillEC(ECArray, nodeEC, sect, counter_tot);
                Level3Utils.fillFTOF(FTOFArray, nodeFTOF, sect, counter_tot);
                Level3Utils.fillHTCC(HTCCArray, nodeHTCC, sect, counter_tot);

                OUTArray.putScalar(new int[] { counter_tot, 0 }, classs);
                OUTArray.putScalar(new int[] { counter_tot, 1 }, p);
                OUTArray.putScalar(new int[] { counter_tot, 2 }, theta);
                OUTArray.putScalar(new int[] { counter_tot, 3 }, sect);
                OUTArray.putScalar(new int[] { counter_tot, 4 }, MC_PID);
                OUTArray.putScalar(new int[] { counter_tot, 5 }, phi);
                OUTArray.putScalar(new int[] { counter_tot, 6 }, nphe);
                OUTArray.putScalar(new int[] { counter_tot, 7 }, PID);
                OUTArray.putScalar(new int[] { counter_tot, 8 }, SF);
                OUTArray.putScalar(new int[] { counter_tot, 9 }, ECAL_e);
                counter_tot++;

            }
            nbread++;
        }

        // System.out.print(OUTArray);
        // System.out.print(DCArray);
        // System.out.print(ECArray);

        MultiDataSet dataset = new MultiDataSet(new INDArray[] { DCArray, ECArray, FTOFArray, HTCCArray }, new INDArray[] { OUTArray });

        INDArray[] outputs = network.output(dataset.getFeatures()[0], dataset.getFeatures()[1], dataset.getFeatures()[3]);

        System.out.printf("\nclass %f @ resp %f\n",OUTArray.getFloat(0,0),outputs[0].getFloat(0, elClass));
        System.out.printf("PID %f, MC PID %f\n", OUTArray.getFloat(0, 7),OUTArray.getFloat(0, 4));
        System.out.printf("P %f, theta %f, phi %f\n",OUTArray.getFloat(0, 1),OUTArray.getFloat(0, 2),OUTArray.getFloat(0, 5));
        System.out.printf("nphe %f, sect %f\n",OUTArray.getFloat(0, 6),OUTArray.getFloat(0, 3));
        System.out.printf("SF %f, ECAL energy %f\n",OUTArray.getFloat(0, 8),OUTArray.getFloat(0, 9));
        plotDCExamples(DCArray, 1, 0);
        plotECExamples(ECArray, 1, 0);
        plotECExamples(HTCCArray, 1, 0);

        
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

    
    public static void main(String[] args){        
        
        String file = "/Users/tyson/data_repo/trigger_data/sims/claspyth_train/clasdis_51.hipo";

        Level3Tester_SingleEvent_SimulationSIDIS t=new Level3Tester_SingleEvent_SimulationSIDIS();
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
        t.test(file,10.6,19,1,elClass,elLabelVal);

        
    }
}
