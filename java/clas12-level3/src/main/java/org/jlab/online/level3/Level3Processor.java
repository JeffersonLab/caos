/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.online.level3;

import j4np.data.base.DataFrame;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.io.HipoReader;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.jlab.online.trainer.Level3Trainer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @authors gavalian, tyson
 */
public class Level3Processor  {
    
    ComputationGraph  network;
    CompositeNode     dcBank = null;
    CompositeNode     ecBank = null;    
    
    public Level3Processor(){
        dcBank = new CompositeNode(11,1,"bbii",1500);
        ecBank = new CompositeNode(12,1,"bbii",1500);
    }
    /**
     * Loads the network from the saved weights.
     *
     * @param url
     */
    public void initNetwork(String url){
        /*try {
            network = KerasModelImport.importKerasModelAndWeights(url);            
            System.out.println(network.summary());            
        } catch (IOException e) {
            System.out.println("IO Exception");
            e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
            System.out.println("Invalid Keras Config");
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            System.out.println("Unsupported Keras Config");
            e.printStackTrace();
        }*/
    }
    
    public void load(String networkFile){
        try {
            network = ComputationGraph.load(new File(networkFile), true);
            System.out.println(network.summary());
            System.out.println("sucessfully loaded the network : " + networkFile);
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    
    public void benchmark(String file, int iterations){
        int[] quantity = new int[]{8,16,32,64,128,512,1024,2048,4096};
        double[] rate = new double[quantity.length];
        
        for(int k = 0; k < quantity.length; k++){
            DataFrame<Event> frame = new DataFrame<>();
            HipoReader r = new HipoReader();
            r.setDebugMode(0);
            r.open(file);
            for(int n = 0; n < quantity[k]; n++) frame.addEvent(new Event());
            r.nextFrame(frame);
            INDArray[] input = Level3Utils.createData(frame.getList());
            
            long then = System.currentTimeMillis();
            System.out.printf(">>>> batch size = %d, n-iterations = %d\n",
                    frame.getList().size(),iterations);
            for(int it = 0; it < iterations; it++){
                INDArray[] output = network.output(input);                
            }
            long now = System.currentTimeMillis();
            double time = (now - then)/1000.0;
            rate[k] = iterations*frame.getList().size()/time;
            System.out.printf(">>>> processing rate = %9.4f evt/sec, time elapsed = %d msec\n",
                    rate[k],now-then);
        }
        
        for(int k = 0; k < quantity.length; k++){
            System.out.printf(">>>> processing rate: batch size = %12d %9.4f evt/sec\n",
                    quantity[k],rate[k]);
        }
        
    }
    

    
    public static void main(String[] args){
        String file  = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.h5_000000_daq.h5";
        String network = "etc/networks/network-level3-dl4j.network";
        if(args.length>0) file = args[0];
        if(args.length>1) network = args[1];
        
        Level3Processor processor = new Level3Processor();
        processor.load(network);
        
        System.out.println("------ starting the level-3 benchmark with file = " + file);
        processor.benchmark(file, 500);
        //processor.initNetwork("etc/networks/network_rgb_50nA_inbending.h5");
        
    }
}
