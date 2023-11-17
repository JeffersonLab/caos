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
public class Level3PyWriter {

    ComputationGraph network = null;

    public Level3PyWriter() {

    }

   
    public static void writeFiles(List<String> names,String dir, String out, List<Integer> pids, List<Integer> Sectors){
        
        int nFile=0;

        for (String name : names) {

            System.out.println("Part Type: "+name);

            String file_in =dir+name+"_rec.hipo" ;

            HipoReader r = new HipoReader(file_in);
            Event e = new Event();

            int nMax = r.entries();

            INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
            INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
            INDArray OUTArray = Nd4j.zeros(nMax, 4);

            Bank[] banks = r.getBanks("DC::tdc", "ECAL::adc", "RUN::config");
            Bank[] dsts = r.getBanks("REC::Particle", "REC::Track", "REC::Calorimeter", "REC::Cherenkov",
                    "ECAL::clusters");

            CompositeNode nodeDC = new CompositeNode(12, 1, "bbsbil", 4096);
            CompositeNode nodeEC = new CompositeNode(11, 2, "bbsbifs", 4096);

            int counter = 0;

            while (r.hasNext() && counter < nMax) {

                r.nextEvent(e);
                e.read(banks);

                e.read(dsts);

                List<Level3Particle> particles = new ArrayList<Level3Particle>();

                // find and initialise particles
                for (int i = 0; i < dsts[0].getRows(); i++) {
                    Level3Particle part = new Level3Particle();
                    part.read_Particle_Bank(i, dsts[0]);
                    part.read_Cal_Bank(dsts[2]);
                    part.read_HTCC_bank(dsts[3]);
                    part.find_sector_cal(dsts[2]);
                    particles.add(part);
                }


                //loop over sectors
                for (int sect: Sectors) {

                    double p=0;
                    Boolean keepEvent=false;
                    double[] vars= new double[] {0,0,0,0};
                    //keep sectors with at least one particle
                    for (Level3Particle part:particles){
                        if(part.Sector==sect){
                            //some fiducial cuts if needed
                            //if(part.check_Energy_Dep_Cut()==true && part.check_FID_Cal_Clusters(dsts[4])==true && part.check_SF_cut()==true){
                            if(part.P>0){
                                keepEvent=true;
                                p=part.P;
                                vars[0]=part.Px;
                                vars[1]=part.Py;
                                vars[2]=part.Pz;
                                vars[3]=part.getM(pids.get(nFile));
                                //System.out.printf("M: %f",vars[3]);
                            }
                        }
                    }

                    Level3Converter_MultiClass.convertDC(banks[0], nodeDC, sect);
                    double nEdep=Level3Converter_MultiClass.convertEC(banks[1], nodeEC, sect);
                    // nodeDC.print();
                    // nodeEC.print();

                    Level3Utils.fillDC(DCArray, nodeDC, sect, counter);
                    Level3Utils.fillEC_noNorm(ECArray, nodeEC, sect, counter);

                    INDArray EventDCArray = DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),
                            NDArrayIndex.all(), NDArrayIndex.all());
                    INDArray EventECArray = ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(),
                            NDArrayIndex.all(), NDArrayIndex.all());

                    // && hasL1==1
                    // check that the images aren't all empty, allow empty DC for neutrals
                    if (EventECArray.any() && keepEvent==true) { // EventDCArray.any()

                        OUTArray.putScalar(new int[] { counter, 0 }, vars[0]);
                        OUTArray.putScalar(new int[] { counter, 1 }, vars[1]);
                        OUTArray.putScalar(new int[] { counter, 2 }, vars[2]);
                        OUTArray.putScalar(new int[] { counter, 3 }, vars[3]);
                        counter++;
                    } else {
                        // erase last entry
                        DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 112));
                        ECArray.get(NDArrayIndex.point(counter), NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all()).assign(Nd4j.zeros(1, 6, 72));
                    }
                }
            }
            File fileEC = new File(out + "EC_" + name + ".npy");
            File fileDC = new File(out + "DC_" + name + ".npy");
            File fileOut = new File(out + "Vars_" + name + ".npy");
            try {
                Nd4j.writeAsNumpy(ECArray, fileEC);
                Nd4j.writeAsNumpy(DCArray, fileDC);
                Nd4j.writeAsNumpy(OUTArray, fileOut);
            } catch (IOException eo) {
                System.out.println("Could not write file");
            }
            nFile++;
        }
    }
    
    public static void main(String[] args){        
        
        String dir = "/Users/tyson/data_repo/trigger_data/sims/";
        String out = "/Users/tyson/data_repo/trigger_data/sims/python/";

        List<String> names = new ArrayList<>();
        names.add("pim");
        names.add("gamma");
        names.add("el" );

        List<Integer> pids = new ArrayList<>();
        pids.add(211);
        pids.add(22);
        pids.add(-11);

        List<Integer> sectors=new ArrayList<Integer>(); //simulated only in sectors 1 and 6
        sectors.add(1);
        sectors.add(6);

        Level3PyWriter pyw=new Level3PyWriter();

        Level3PyWriter.writeFiles(names,dir,out,pids,sectors);
        
    }
}
