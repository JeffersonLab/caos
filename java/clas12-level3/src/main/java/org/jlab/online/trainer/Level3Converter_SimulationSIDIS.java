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
public class Level3Converter_SimulationSIDIS {

    ComputationGraph network = null;

    public Level3Converter_SimulationSIDIS() {

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

    public static void convertData(String dir,double beamE, int desiredPID,String out,int nFiles){
        int nPart=0;

        HipoReader r = new HipoReader(dir + "clasdis_1.hipo");
        HipoWriter w = HipoWriter.create(out,r);
        Event e_out = new Event();

        int start=1;

        while (start < nFiles) {//56

            String file = dir + "clasdis_" + String.valueOf(start) + ".hipo";
            //String file=dir;
            start++;

            r = new HipoReader(file);
            
            Event e = new Event();

            // r.getSchemaFactory().show();

            CompositeNode nodeDC = new CompositeNode(12, 1, "bbsbil", 4096);
            CompositeNode nodeEC = new CompositeNode(11, 2, "bbsbifs", 4096);
            CompositeNode nodeFTOF = new CompositeNode(13, 3, "bbsbifs", 4096);
            CompositeNode nodeHTCC = new CompositeNode(14, 5, "bbsbifs", 4096);

            Bank[] banks = r.getBanks("DC::tdc", "ECAL::adc", "RUN::config", "FTOF::adc", "HTCC::adc");
            Bank[] dsts = r.getBanks("REC::Particle", "REC::Track", "REC::Calorimeter", "REC::Cherenkov",
                    "ECAL::clusters", "MC::Particle");

            while (r.hasNext()) {

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

                List<Level3Particle> particles = new ArrayList<Level3Particle>();

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
                        particles.add(part);
                    }
                }

                // loop over sectors
                for (int sect = 1; sect < 7; sect++) {

                    double p = 0;
                    Boolean hasDesiredParticle = false, hasEl=false;

                    //check if we have at least one particle
                    //and if there's an electron in the sector
                    for (Level3Particle part : particles) {
                        if (part.Cal_Sector == sect ){//&& nPart_pSect(particles, sect)==1) { //&& nTrack_pSect(dsts[1],sect)==1
                            if (part.TruthMatch(0.1, 0.1, 0.1)) {
                                if (part.P > 0.5) {
                                    if (part.MC_PID == 11) {
                                        if (part.check_Energy_Dep_Cut() == true
                                                && part.check_FID_Cal_Clusters(dsts[4]) == true
                                                && part.check_SF_cut() == true 
                                                && part.Nphe>=2) {
                                            hasEl = true;
                                            if (part.MC_PID == desiredPID) {
                                                hasDesiredParticle = true;
                                                p = part.P;
                                            } 
                                        }
                                    } else if (part.MC_PID == -11) {
                                        if (part.check_Energy_Dep_Cut() == true
                                                && part.check_FID_Cal_Clusters(dsts[4]) == true) {
                                            if (part.MC_PID == desiredPID) {
                                                hasDesiredParticle = true;
                                                p = part.P;
                                            } 
                                        }
                                    } else if (part.MC_PID == 22) {
                                        //tighter cuts on photons
                                        //also only want photons on their own
                                        //not photons produced by other particles
                                        if (part.P > 0 && part.TruthMatch(0.5, 0.5, 0.5) && nPart_pSect(particles, sect)==1) {
                                            if (part.MC_PID == desiredPID) {
                                                hasDesiredParticle = true;
                                                p = part.P;
                                            } 
                                        }
                                    } else {
                                        if (part.MC_PID == desiredPID) {
                                            hasDesiredParticle = true;
                                            p = part.P;
                                        } 
                                    }
                                }
                            }
                        } 
                    }

                    //if no particle in sector then empty
                    // PID==0 calls for empty
                    if(nPart_pSect(particles, sect)==0 && desiredPID==0){
                        hasDesiredParticle=true;
                    }

                    //if an event has an electron
                    //don't care if it has other particles
                    //we want it to be in electron sample
                    Boolean keepEvent=false;
                    if (hasDesiredParticle) {
                        if (desiredPID==11) {
                            keepEvent=true;
                        }else{
                            if (!hasEl) {
                                keepEvent=true;
                            }
                        }
                    }

                    int[] labels = new int[] { desiredPID, Level3Converter_MultiClass.getPTag(p), sect };
                    Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                    Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                    Level3Converter_MultiClass.convertFTOF(banks[3], nodeFTOF, sect);
                    Level3Converter_MultiClass.convertHTCC(banks[4], nodeHTCC, sect);
                    // nodeDC.print();
                    // nodeEC.print();
                    Node tnode = new Node(5, 4, labels);

                    if ( nodeEC.getRows() > 0 && keepEvent) {//nodeDC.getRows() > 0

                        // all data
                        e_out.reset();
                        e_out.write(nodeEC);
                        e_out.write(nodeDC);
                        e_out.write(nodeHTCC);
                        e_out.write(nodeFTOF);
                        e_out.write(tnode);

                        w.addEvent(e_out);
                        nPart++;
                        
                    }
                }
            }
        }
        // System.out.print(OUTArray);
        // System.out.print(DCArray);
        // System.out.print(ECArray);
        System.out.printf("PID %d, nEvents %d\n\n", desiredPID,nPart);

        w.close();

    }

    
    
    public static void main(String[] args){        
        
        String dir = "/Users/tyson/data_repo/trigger_data/sims/claspyth_train/";


        String[] part={"el","pim","gamma","pos","pip","p","empty"};//,"pi0","mup","mum"};
        Integer[] pid={11,-211,22,-11,211,2212,0};

        for(int i=0;i<7;i++){
            int nFiles=56;
            //lots of empty events so don't want to create huge file
            if(pid[i]==0){
                nFiles=5;
            }
            Level3Converter_SimulationSIDIS.convertData(dir,10.6,pid[i],dir+"claspyth_"+part[i]+"_daq.h5",nFiles);//10000
        }
       
        
    }
}
