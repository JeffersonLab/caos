package org.jlab.online.trainer;

import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;
import twig.data.GraphErrors;
import twig.graphics.TGCanvas;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;

public class Level3SampleEfficiency_Simulation {

    public static INDArray calcSelectionEfficiency(String file, int pid,List<Integer> Sectors,int nMax){
        HipoReader r = new HipoReader(file);
        Event e = new Event();
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config","FTOF::adc","HTCC::adc");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter","REC::Cherenkov","ECAL::clusters","MC::Particle");

        INDArray OUTArray = Nd4j.zeros(nMax, 3);
        
        double n=0,mip=0;
        while(r.hasNext() && n<nMax){
            
            r.nextEvent(e);
            e.read(banks);
            e.read(dsts);

            /*System.out.println("REC::Particle");
            dsts[0].show();
            System.out.println("REC::Calorimeter");
            dsts[2].show();
            System.out.println("ECAL::clusters");
            dsts[4].show();*/

            List<Level3Particle> particles = new ArrayList<Level3Particle>();

            //find and initialise particles
            for(int i=0;i<dsts[0].getRows();i++){
                Level3Particle part=new Level3Particle();
                part.read_Particle_Bank(i,dsts[0]);
                part.read_MCParticle_Bank(0, dsts[5]);
                part.read_Cal_Bank(dsts[2]);
                part.read_HTCC_bank(dsts[3]);
                //part.find_sector_track(dsts[1]);
                part.find_sector_cal(dsts[2]); //use cal for neutrals
                particles.add(part);
            }

            //loop over sectors
            for (int sect: Sectors) {

                double theta = 0;
                double p=0;
                //keep sectors with at least one particle
                for (Level3Particle part:particles){
                    if(part.Sector==sect){
                        
                        if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                            int selected=0;
                            if(part.isMip(dsts[4])){
                                selected=1;
                                mip++;
                            }
                            p = part.P;
                            theta = part.Theta * (180.0 / Math.PI);
                            OUTArray.putScalar(new int[] {(int) n, 0 }, selected);
                            OUTArray.putScalar(new int[] {(int) n, 1 }, p);
                            OUTArray.putScalar(new int[] {(int) n, 2 }, theta);
                            n++;
                        }
                        
                    }
                }
            }
        }

        double avEff=mip/n;
        System.out.printf("\n PID %d , Average Efficiency %f\n\n",pid,avEff);

        return OUTArray;
    }

    public static void plotVarDep(List<INDArray> AllLabels,List<String> partNames,int cutVar, String varName, String varUnits,double low, double high,double step) {

        TGCanvas c = new TGCanvas();
        c.setTitle("Efficiency vs "+varName);

        for (int j = 0; j < AllLabels.size(); j++) {
            INDArray Labels=AllLabels.get(j);

            GraphErrors gEff = new GraphErrors();
            gEff.attr().setMarkerColor(j+2);
            gEff.attr().setMarkerSize(10);
            gEff.attr().setTitle(partNames.get(j));
            gEff.attr().setTitleX(varName + " " + varUnits);
            gEff.attr().setTitleY("Efficiency");


            for (double q2 = low; q2 < high; q2 += step) {
                double n = 0, nPass = 0;
                for (int i = 0; i < Labels.shape()[0]; i++) {
                    if (Labels.getFloat(i, cutVar) > low && Labels.getFloat(i, cutVar) < high) {
                        n++;
                        if (Labels.getFloat(i, 0) == 1) {
                            nPass++;
                        }
                    }
                }
                gEff.addPoint(q2 + step / 2, nPass / n, 0, 0);

            } 

            if (j == 0) {
                c.draw(gEff);
            } else {
                c.draw(gEff, "same");
            }
        }
        
        c.region().axisLimitsY(-0.05, 1.05);
        c.region().showLegend(0.6, 0.5);

    }

    public static void main(String[] args){      

        String dir="/Users/tyson/data_repo/trigger_data/sims/"; //_AI

        //String[] base={"el","pim","gamma","pos","pip","p","pi0","mup","mum"};
        //int[] pid={11,-211,22,-11,211,2212,111,13,-13};

        String[] base={"pim","pip","mum","mup"};
        int[] pid={-211,211,13,-13};

        List<Integer> sectors=new ArrayList<Integer>(); //simulated only in sectors 1 
        sectors.add(1);

        List<INDArray> OUTArrays=new ArrayList<INDArray>();
        List<String> ParticleNames=new ArrayList<String>();
        ParticleNames.add("pi-");
        ParticleNames.add("pi+");
        ParticleNames.add("mu-");
        ParticleNames.add("mu+");

        for (int file=0;file<4;file+=1){
            String fName=dir+base[file]+"_rec.hipo";
            INDArray OUTArray=Level3SampleEfficiency_Simulation.calcSelectionEfficiency(fName,pid[file],sectors,10000); //_wrongTriggerOnly
            OUTArrays.add(OUTArray);
        }

        plotVarDep(OUTArrays,ParticleNames,1, "Momentum", "[GeV]",1,9.0,1.0);
        plotVarDep(OUTArrays,ParticleNames,2, "Theta", "[Deg]",10.0,35.0,5.);


    }
    
}
