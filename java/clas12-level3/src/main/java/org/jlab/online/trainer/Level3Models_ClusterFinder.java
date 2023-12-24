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
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.lossfunctions.impl.LossMAE;

/**
 *
 * @author gavalian, tyson
 */
public class Level3Models_ClusterFinder {

    public static ComputationGraphConfiguration getModel0a(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(6,4)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(6,4)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,1).build()
                        , "L1")            
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{3,3}).poolingType(PoolingType.MAX).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().activation(Activation.RELU).nOut(300).dropOut(0.2).build(), "L1Pool")
                .addLayer("dcDense2", new DenseLayer.Builder().activation(Activation.RELU).nOut(200).dropOut(0.2).build(), "dcDense")
                .addLayer("out", new OutputLayer.Builder(new LossBinaryXENT())//new LossMSE()
                        .nIn(200).nOut(108)
                        .activation(Activation.SIGMOID)
                        .build()
                        , "dcDense2")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(36, 112, 1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getModel0b(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc","ftof")
                .addLayer("L1", new ConvolutionLayer.Builder(6,4)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(6,4)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,1).build()
                        , "L1")            
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{3,3}).poolingType(PoolingType.MAX).build(), "L2")
                .addLayer("dcDense",
                        new DenseLayer.Builder().activation(Activation.RELU).nOut(300).dropOut(0.2).build(), "L1Pool")
                .addLayer("L1ftof", new ConvolutionLayer.Builder(6, 1)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(2, 1).build(), "ftof")
                .addLayer("ftofDense", new DenseLayer.Builder().nOut(100).dropOut(0.5).build(), "L1ftof")
                .addVertex("merge", new MergeVertex(), "dcDense","ftofDense")
                .addLayer("dense1", new DenseLayer.Builder().nOut(200).dropOut(0.5).build(), "merge")
                .addLayer("out", new OutputLayer.Builder(new LossBinaryXENT())//new LossMSE()
                        .nIn(200).nOut(108)
                        .activation(Activation.SIGMOID)
                        .build()
                        , "dense1")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(36, 112, 1),InputType.convolutional(62,1,1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getModel0c(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc","ftof")
                .addLayer("L1", new ConvolutionLayer.Builder(6,4)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,2).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(3, 2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(2, 2).build(), "L1")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L2")
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L3")
                .addLayer("L5", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L4")
                .addLayer("L6", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L5")
                .addLayer("L7", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L6")
                .addLayer("L8", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L7")
                .addLayer("L1Pool",
                        new SubsamplingLayer.Builder(new int[] { 1, 2 }, new int[] { 1, 2 })
                                .poolingType(PoolingType.MAX).build(),
                        "L8")
                .addLayer("dcDense",
                        new DenseLayer.Builder().activation(Activation.RELU).nOut(300).dropOut(0.2).build(), "L1Pool")
                .addLayer("L1ftof", new ConvolutionLayer.Builder(6, 1)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(2, 1).build(), "ftof")
                .addLayer("ftofDense", new DenseLayer.Builder().nOut(100).dropOut(0.5).build(), "L1ftof")
                .addVertex("merge", new MergeVertex(), "dcDense","ftofDense")
                .addLayer("dense1", new DenseLayer.Builder().nOut(200).dropOut(0.5).build(), "merge")
                .addLayer("out", new OutputLayer.Builder(new LossBinaryXENT())//new LossMSE()
                        .nIn(200).nOut(108)
                        .activation(Activation.SIGMOID)
                        .build()
                        , "dense1")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(36, 112, 1),InputType.convolutional(62,1,1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getModel0d(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(3,4)
                        .nIn(18)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "L1")            
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{1,3}, new int[]{1,3}).poolingType(PoolingType.MAX).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().activation(Activation.RELU).nOut(300).dropOut(0.2).build(), "L1Pool")
                .addLayer("dcDense2", new DenseLayer.Builder().activation(Activation.RELU).nOut(200).dropOut(0.2).build(), "dcDense")
                .addLayer("out", new OutputLayer.Builder(new LossBinaryXENT())//new LossMSE()
                        .nIn(200).nOut(108)
                        .activation(Activation.SIGMOID)
                        .build()
                        , "dcDense2")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 6))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getModel0e(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                //.l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))//Adam(5e-3) //AdaDelta()
                .graphBuilder()
                .addInputs("dc","ecin")
                .addLayer("L1", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(3,4)
                        .nIn(18)
                        .nOut(18)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "L1")            
                .addLayer("L1Pool", new SubsamplingLayer.Builder(new int[]{1,3}, new int[]{1,3}).poolingType(PoolingType.MAX).build(), "L2")
                .addLayer("dcDense", new DenseLayer.Builder().activation(Activation.RELU).nOut(300).dropOut(0.2).build(), "L1Pool")
                .addLayer("L1ecin", new ConvolutionLayer.Builder(1, 9)
                        .nIn(1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1, 2).build(), "ecin")
                .addLayer("ecinDense", new DenseLayer.Builder().nOut(100).dropOut(0.5).build(), "L1ecin")
                .addVertex("merge", new MergeVertex(), "dcDense","ecinDense")
                .addLayer("Dense", new DenseLayer.Builder().activation(Activation.RELU).nOut(200).dropOut(0.2).build(), "merge")
                .addLayer("out", new OutputLayer.Builder(new LossBinaryXENT())//new LossMSE()
                        .nIn(200).nOut(108)
                        .activation(Activation.SIGMOID)
                        .build()
                        , "Dense")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(6, 112, 6),InputType.convolutional(1,108,1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getTestLayerL1(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(6)                 
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "dc")
                .setOutputs("L1")
                .setInputTypes(InputType.convolutional(6, 112, 6))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getTestLayerL2(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(3,4)
                        .nIn(6)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .padding(1,1)
                        .stride(1,1).build()
                        , "L1")       
                .setOutputs("L2")
                .setInputTypes(InputType.convolutional(6, 112, 6))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getTestLayerL4(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(6,4)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,2).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(6, 4)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(2, 2).build(), "L1")
                .addLayer("L3", new ConvolutionLayer.Builder(3,3)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L2")
                .addLayer("L4", new ConvolutionLayer.Builder(3,3)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L3")
                .setOutputs("L4")
                .setInputTypes(InputType.convolutional(36, 112, 1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }

    public static ComputationGraphConfiguration getTestLayerL8(){
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("dc")
                .addLayer("L1", new ConvolutionLayer.Builder(6,4)
                        .nIn(1)
                        .nOut(6)                    
                        .activation(Activation.RELU)
                        .stride(2,2).build()
                        , "dc")
                .addLayer("L2", new ConvolutionLayer.Builder(3, 2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(2, 2).build(), "L1")
                .addLayer("L3", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L2")
                .addLayer("L4", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L3")
                .addLayer("L5", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L4")
                .addLayer("L6", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L5")
                .addLayer("L7", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L6")
                .addLayer("L8", new ConvolutionLayer.Builder(2,2)
                        .nIn(6)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .stride(1,1).build(), "L7")
                .setOutputs("L8")
                .setInputTypes(InputType.convolutional(36, 112, 1))
                .build();
                //config.setValidateOutputLayerConfig(false);
        return config;
    }
    
    public static ComputationGraphConfiguration getModel(String modelname){
        switch(modelname){
            case "0a": return Level3Models_ClusterFinder.getModel0a();
            case "0b": return Level3Models_ClusterFinder.getModel0b();
            case "0c": return Level3Models_ClusterFinder.getModel0c();
            case "0d": return Level3Models_ClusterFinder.getModel0d();
            case "0e": return Level3Models_ClusterFinder.getModel0e();
            case "testL1": return Level3Models_ClusterFinder.getTestLayerL1();
            case "testL2": return Level3Models_ClusterFinder.getTestLayerL2();
            case "testL4": return Level3Models_ClusterFinder.getTestLayerL4();
            default: return Level3Models_ClusterFinder.getModel0a();
        }
    }
}
