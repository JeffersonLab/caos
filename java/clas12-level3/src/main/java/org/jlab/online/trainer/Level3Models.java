/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 *
 * @author gavalian
 */
public class Level3Models {
    
    public static ComputationGraphConfiguration getModel0a(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))                
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")
                .addLayer("dcDense", new DenseLayer.Builder().nIn(3330).nOut(48).dropOut(0.5).build(), "L1")
                .addLayer("ecDense", new DenseLayer.Builder().nIn(2130).nOut(48).dropOut(0.5).build(), "L2")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(48+48).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    
    public static ComputationGraphConfiguration getModel0b(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))                
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L1")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L1Pool")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L2Pool")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(48+48).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }


    public static ComputationGraphConfiguration getModel0bw(){

        //weighting, class 1 twice as important
        INDArray weightsArray = Nd4j.create(new double[]{0.5, 1.0});

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))                
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L1")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L1Pool")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L2Pool")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(new LossMCXENT(weightsArray))     // *** Weighted loss function configured here ***
                        .nIn(48+48).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    public static ComputationGraphConfiguration getModel0c(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))                
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L1")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L1Pool")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L2Pool")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
                .addLayer("hidden", new DenseLayer.Builder().nOut(12).dropOut(0.5).build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(12).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "hidden")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    public static ComputationGraphConfiguration getModel(String modelname){
        switch(modelname){
            case "0a": return Level3Models.getModel0a();
            case "0b": return Level3Models.getModel0b();
            case "0bw": return Level3Models.getModel0bw();
            case "0c": return Level3Models.getModel0c();
            default: return null;
        }
    }
}
