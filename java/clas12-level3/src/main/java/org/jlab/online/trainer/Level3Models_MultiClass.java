/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1D;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
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
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
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
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
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

    public static ComputationGraphConfiguration getModel0d_FTOFHTCC(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec","ftof","htcc")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L2")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L3")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L4")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L1Pool")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(48).dropOut(0.5).build(), "L2Pool")
                .addLayer("L1ftof", new ConvolutionLayer.Builder(4,1)
                            .nIn(1)
                            .nOut(6)
                            .activation(Activation.RELU)
                            .stride(2,1).build()
                            ,"ftof")
                .addLayer("ftofDense", new DenseLayer.Builder().nOut(30).dropOut(0.5).build(), "L1ftof")
                .addLayer("L1htcc", new ConvolutionLayer.Builder(2,1)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        ,"htcc")
                .addLayer("htccDense", new DenseLayer.Builder().nOut(20).dropOut(0.5).build(), "L1htcc")
                .addVertex("merge", new MergeVertex(), "dcDense", "ecDense","ftofDense","htccDense")
                .addLayer("dense1", new DenseLayer.Builder().nOut(100).dropOut(0.5).build(), "merge")
                .addLayer("dense2", new DenseLayer.Builder().nOut(25).dropOut(0.5).build(), "dense1")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(25).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "dense2")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 1), InputType.convolutional(6, 72, 1),
                                InputType.convolutional(62,1,1), InputType.convolutional(8,1,1))
                .build();
        return config;
    }

    public static ComputationGraphConfiguration getModel0d_allLayers(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L2")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{12,2}, new int[]{12,2}).build(), "L3")
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
                .setInputTypes(InputType.convolutional(36, 112, 1),InputType.convolutional(6, 72, 1))
                .build();
        return config;
    }

    public static ComputationGraphConfiguration getModel0e(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec")
                .addLayer("L1", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(2)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L5", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L3")
                .addLayer("L7", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)                    
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L5")
                .addLayer("L2", new ConvolutionLayer.Builder(2,2)
                        .nIn(1)
                        .nOut(2)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L2")
                .addLayer("L6", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L4") 
                        .addLayer("L8", new ConvolutionLayer.Builder(2,2)
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L6")               
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L7")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{2,2}, new int[]{2,2}).build(), "L8")
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

    public static ComputationGraphConfiguration getModel0f(int nClasses){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc", "ec","htcc")
                .addLayer("L1", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L3", new ConvolutionLayer.Builder(3,4)
                        .nIn(18)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "L1")
                .addLayer("L2", new ConvolutionLayer.Builder(3,6)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "ec")    
                .addLayer("L4", new ConvolutionLayer.Builder(3,6)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        , "L2")                
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{1,3}, new int[]{1,3}).poolingType(PoolingType.MAX).build(), "L3")
                .addLayer("L2Pool", new SubsamplingLayer.Builder(new int[]{1,2}, new int[]{1,2}).build(), "L4")
                .addLayer("dcDense", new DenseLayer.Builder().nOut(300).dropOut(0.5).build(), "L1Pool")
                .addLayer("dcDense2", new DenseLayer.Builder().nOut(200).dropOut(0.5).build(), "dcDense")
                .addLayer("ecDense", new DenseLayer.Builder().nOut(200).dropOut(0.5).build(), "L2Pool")
                .addLayer("L1htcc", new ConvolutionLayer.Builder(2,1)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build()
                        ,"htcc")
                .addLayer("htccDense", new DenseLayer.Builder().nOut(40).dropOut(0.5).build(), "L1htcc")
                .addVertex("merge", new MergeVertex(), "dcDense2", "ecDense","htccDense")
                .addLayer("dense1", new DenseLayer.Builder().nOut(200).dropOut(0.5).build(), "merge")
                .addLayer("dense2", new DenseLayer.Builder().nOut(100).dropOut(0.5).build(), "dense1")
                .addLayer("out", new OutputLayer.Builder()
                        .nIn(100).nOut(nClasses)
                        .activation(Activation.SOFTMAX)
                        .build()
                        , "dense2")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 6), InputType.convolutional(6, 72, 1), InputType.convolutional(8,1,1))
                .build();
        return config;
    }
    
    public static ComputationGraphConfiguration getModel(String modelname, int nClasses){
        switch(modelname){
            case "0a": return Level3Models_MultiClass.getModel0a(nClasses);
            case "0b": return Level3Models_MultiClass.getModel0b(nClasses);
            case "0c": return Level3Models_MultiClass.getModel0c(nClasses);
            case "0d": return Level3Models_MultiClass.getModel0d(nClasses);
            case "0d_FTOFHTCC": return Level3Models_MultiClass.getModel0d_FTOFHTCC(nClasses);
            case "0d_allLayers": return Level3Models_MultiClass.getModel0d_allLayers(nClasses);
            case "0e": return Level3Models_MultiClass.getModel0e(nClasses);
            case "0f": return Level3Models_MultiClass.getModel0f(nClasses);
            default: return null;
        }
    }
}
