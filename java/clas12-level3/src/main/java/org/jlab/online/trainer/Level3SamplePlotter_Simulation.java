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
public class Level3SamplePlotter_Simulation {

    

    public Level3SamplePlotter_Simulation() {

    }

    public static H1F makeNiceHist(String title, String axis, int Color, int nBins, double low, double high) {
        H1F h = new H1F(title, nBins, low, high);
        h.attr().setLineColor(Color);
        //h.attr().setFillColor(2);
        h.attr().setLineWidth(3);
        h.attr().setTitleX(axis);
        return h;
    }

    public static H2F makeNice2DHist(String title, String axis, int nBins, double low, double high,String axis2,int nBins2, double low2, double high2) {
        H2F h = new H2F(title, nBins, low, high, nBins2, low2, high2);
        h.attr().setTitleX(axis);
        h.attr().setTitleY(axis2);
        return h;
    }

    
   
    public static void plot(List<String> files,List<Integer> pids,List<Integer> Sectors,int max){

        H1F elP=makeNiceHist("e-","Momentum [GeV]",2,110,0,11);
        H1F poP=makeNiceHist("e+","Momentum [GeV]",3,110,0,11);
        H1F gammaP=makeNiceHist("photon","Momentum [GeV]",5,110,0,11);
        H1F pimP=makeNiceHist("pi-","Momentum [GeV]",6,110,0,11);

        H1F elP_gen=makeNiceHist("e- (truth)","Momentum [GeV]",2,110,0,11);
        H1F poP_gen=makeNiceHist("e+ (truth)","Momentum [GeV]",3,110,0,11);
        H1F gammaP_gen=makeNiceHist("photon (truth)","Momentum [GeV]",5,110,0,11);
        H1F pimP_gen=makeNiceHist("pi- (truth)","Momentum [GeV]",6,110,0,11);

        H2F elPTheta=makeNice2DHist("e-","e- Momentum [GeV]",110,0,11,"e- Theta [Deg]",100,0,40);
        H2F poPTheta=makeNice2DHist("e+","e+ Momentum [GeV]",110,0,11,"e+ Theta [Deg]",100,0,40);
        H2F gammaPTheta=makeNice2DHist("Photon","photon Momentum [GeV]",110,0,11,"photon Theta [Deg]",100,0,40);
        H2F pimPTheta=makeNice2DHist("pi-","pi- Momentum [GeV]",110,0,11,"pi- Theta [Deg]",100,0,40);

        int classs=0, nEls = 0, nPosi = 0, nGamma = 0, nPim = 0;;
        for (String file : files) {
            HipoReader r = new HipoReader(file);
            Event e = new Event();

            int nMax = max;
            if (r.entries() < max)
                nMax = r.entries();

            Bank[] banks = r.getBanks("DC::tdc", "ECAL::adc", "RUN::config");
            Bank[] dsts = r.getBanks("REC::Particle", "REC::Track", "REC::Calorimeter", "REC::Cherenkov","ECAL::clusters","MC::Particle");

            int counter=0;
            while (r.hasNext() && counter < nMax) {

                r.nextEvent(e);
                e.read(banks);
                e.read(dsts);

                //dsts[0].show();

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
                    Boolean keepEvent = false;

                    //fille Gen plots without truthmatching
                    //otherwise they look like rec
                     for (Level3Particle part : particles) {
                        if (pids.get(classs) == 11) {
                            elP_gen.fill(part.MC_P);
                        } else if(pids.get(classs) == -11){
                            poP_gen.fill(part.MC_P);
                        }else if (pids.get(classs) == 22) {
                            gammaP_gen.fill(part.MC_P);
                        }else if (pids.get(classs) == -211) {
                            pimP_gen.fill(part.MC_P);
                        }
                     }
                    
                    // keep sectors with at least one particle
                    for (Level3Particle part : particles) {
                        if (part.Sector == sect) {
                            if (pids.get(classs) == 11) {
                                if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                    nEls++;
                                    elP.fill(part.P);
                                    elPTheta.fill(part.P, part.Theta * (180.0 / Math.PI));
                                    counter++;
                                }
                            } else if (pids.get(classs) == -11) {
                                if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                    nPosi++;
                                    poP.fill(part.P);
                                    poPTheta.fill(part.P, part.Theta * (180.0 / Math.PI));
                                    counter++;
                                }
                            } else if (pids.get(classs) == 22) {
                                if (part.P > 0 && part.TruthMatch(0.5, 0.5, 0.5)) {
                                    nGamma++;
                                    gammaP.fill(part.P);
                                    gammaPTheta.fill(part.P, part.Theta * (180.0 / Math.PI));
                                    counter++;
                                }
                            } else if (pids.get(classs) == -211) {
                                if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                    nPim++;
                                    pimP.fill(part.P);
                                    pimPTheta.fill(part.P, part.Theta * (180.0 / Math.PI));
                                    counter++;
                                }
                            }
                        }
                    }
                }     
                        
            }
            classs++;
        }
        //System.out.print(OUTArray);
        //System.out.print(DCArray);
        //System.out.print(ECArray);
        System.out.printf("nEl %d, nPosi %d, nGamma %d, nPim %d\n\n",nEls,nPosi,nGamma,nPim);
        
        TGCanvas cP = new TGCanvas();
        cP.setTitle("Momentum");
        cP.draw(elP).draw(poP,"same").draw(pimP,"same").draw(gammaP,"same");
        cP.region().showLegend(0.7, 0.95);

        TGCanvas cP_gen = new TGCanvas();
        cP_gen.setTitle("Momentum (truth)");
        cP_gen.draw(elP_gen).draw(poP_gen,"same").draw(pimP_gen,"same").draw(gammaP_gen,"same");
        cP_gen.region().showLegend(0.7, 0.95);

        TGCanvas cPThel = new TGCanvas();
        cPThel.setTitle("e- Momentum v Theta");
        cPThel.draw(elPTheta);

        TGCanvas cPThpo = new TGCanvas();
        cPThpo.setTitle("e+ Momentum v Theta");
        cPThpo.draw(poPTheta);

        TGCanvas cPThpos = new TGCanvas();
        cPThpos.setTitle("pi- Momentum v Theta");
        cPThpos.draw(pimPTheta);

        TGCanvas cPThneg = new TGCanvas();
        cPThneg.setTitle("photon Momentum v Theta");
        cPThneg.draw(gammaPTheta);
    }

    
    
    public static void main(String[] args){        
        String dir = "/Users/tyson/data_repo/trigger_data/sims/";

        List<String> files = new ArrayList<>();
        files.add(dir+"pim_rec.hipo");
        files.add(dir+"gamma_rec.hipo");
        files.add(dir+"el_rec.hipo");
        files.add(dir+"pos_rec.hipo");

        List<Integer> pids=new ArrayList<>();
        pids.add(-211);
        pids.add(22);
        pids.add(11);
        pids.add(-11);

        List<Integer> sectors=new ArrayList<Integer>(); //simulated only in sectors 1 and 6
        sectors.add(1);

        Level3SamplePlotter_Simulation P=new Level3SamplePlotter_Simulation();

        
        Level3SamplePlotter_Simulation.plot(files,pids,sectors,70000);
        
    }
}
