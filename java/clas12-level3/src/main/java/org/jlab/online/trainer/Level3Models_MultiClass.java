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
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

/**
 *
 * @author gavalian, tyson
 */
public class Level3Models_MultiClass {
    
    public static ComputationGraphConfiguration getModel0a(int nClasses){
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
                        .nIn(48+48).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    
    public static ComputationGraphConfiguration getModel0b(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())//Adam(5e-3) //AdaDelta()
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
                        .nIn(48+48).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    public static ComputationGraphConfiguration getModel0c(int nClasses){
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
                        .nIn(12).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "hidden")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }

    public static ComputationGraphConfiguration getModel0d(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(32)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L2")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L3")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L4")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L1Pool")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L2Pool")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(48+48).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "merge")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }
    
    public static ComputationGraphConfiguration getModel(String modelname, int nClasses){
        switch(modelname){
            case "0a": return Level3Models_MultiClass.getModel0a(nClasses);
            case "0b": return Level3Models_MultiClass.getModel0b(nClasses);
            case "0c": return Level3Models_MultiClass.getModel0c(nClasses);
            case "0d": return Level3Models_MultiClass.getModel0c(nClasses);
            default: return null;
        }
    }
}
