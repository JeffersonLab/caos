/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

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
import java.util.List;
import java.lang.Math;

import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian, tyson
 */
public class Level3Converter_Simulation {
    
    public static void convertFile(String file, String output,int pid,int tag, List<Integer> Sectors){
        HipoReader r = new HipoReader(file);
        HipoWriter w = HipoWriter.create(output, r);
        Event e = new Event();
        Event e_out = new Event();
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter","REC::Cherenkov","ECAL::clusters","MC::Particle");
        
        CompositeNode nodeDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nodeEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        int n=0;
        while(r.hasNext()){
            
            r.nextEvent(e);
            e.read(banks);
            
            e.read(dsts);

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

                double p=0;
                Boolean keepEvent=false;
                //keep sectors with at least one particle
                for (Level3Particle part:particles){
                    if(part.Sector==sect){
                        if (pid == 11) {
                            if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                if(part.check_Energy_Dep_Cut()==true && part.check_FID_Cal_Clusters(dsts[4])==true && part.check_SF_cut()==true){
                                    keepEvent=true;
                                    p=part.MC_P;
                                    n++;
                                }
                            }
                        } else if (pid == -11) {
                            if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                keepEvent=true;
                                p=part.MC_P;
                                n++;
                            }
                        } else if (pid == 22) {
                            if (part.P > 0 && part.TruthMatch(0.5, 0.5, 0.5)) {
                                keepEvent=true;
                                p=part.MC_P;
                                n++;
                            }
                        } else if (pid == -211) {
                            if (part.P > 0.5 && part.TruthMatch(1.0, 1.0, 1.0)) {
                                keepEvent=true;
                                p=part.MC_P;
                                n++;
                            }
                        }
                    }
                }

                int[] labels = new int[] { pid, Level3Converter_MultiClass.getPTag(p), sect };
                Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                double nEdep=Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                // nodeDC.print();
                // nodeEC.print();

                //System.out.printf("Tag: (%d)",tag);

                Node tnode = new Node(5, 4, labels);

                if (nodeEC.getRows() > 0  && keepEvent==true) {//&& nodeDC.getRows() > 0 allow DC to be empty for neutrals
                    
                    // all data
                    e_out.reset();
                    e_out.write(nodeEC);
                    e_out.write(nodeDC);
                    e_out.write(tnode);

                    e_out.setEventTag(tag);

                    w.addEvent(e_out);

                }
            }
        }
        w.close();

        
        System.out.printf("Particle Type %d, Number of Events: %d\n\n",pid,n);
    }
    
    public static void main(String[] args){        
        

        String dir="/Users/tyson/data_repo/trigger_data/sims/"; //_AI
        String[] base={"el","pim","gamma","pos"};
        int[] tags={1,2,7,3};
        int[] pid={11,-211,22,-11};//not sure if this is useful yet
        List<Integer> sectors=new ArrayList<Integer>(); //simulated only in sectors 1 and 6
        sectors.add(1);

        //for (int file=100;file<110;file+=5){
        for (int file=0;file<4;file+=1){

            String fName=dir+base[file]+"_rec.hipo";
            String out=dir+base[file]+"_daq.h5";
            Level3Converter_Simulation.convertFile(fName,out,pid[file],tags[file],sectors); //_wrongTriggerOnly

        }

    }
}
