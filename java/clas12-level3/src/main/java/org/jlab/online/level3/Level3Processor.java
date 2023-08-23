/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.jlab.online.level3;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @authors gavalian, tyson
 */
public class Level3Processor  {
    
    double inferenceThreshold = 0.5;
    ComputationGraph network;
    
    /**
     *  Loads the network from the saved weights.
     *
     * @param url
     */
    public void initNetwork(String url){
        try {
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
        }
    }
    
  
    
    public static void main(String[] args){
        Level3Processor processor = new Level3Processor();
        processor.initNetwork("etc/networks/network_rgb_50nA_inbending.h5");
    }
}
