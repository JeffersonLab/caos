/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.io.HipoReader;
import j4np.utils.io.TextFileWriter;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import twig.data.GraphErrors;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Trainer {

    ComputationGraph  network = null;
    public        int nEpochs = 25;
    public    String cnnModel = "0a";
    
    public Level3Trainer(){
        
    }
    
    public void initNetwork(){
        /*
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
                */
        ComputationGraphConfiguration config = Level3Models.getModel(cnnModel);
        network = new ComputationGraph(config);
        network.init();
        System.out.println(network.summary());        
    }
    
    public void save(String file){
        
        try {
            network.save(new File(file+"_"+cnnModel+".network"));
            System.out.println("saved file : " + file +"\n"+"done...");
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void load(String file){
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void evaluateFile(String file, int nEvents){
        INDArray[] inputs  = this.getFromFile(file, nEvents);
        INDArray[] outputs = network.output(inputs[0],inputs[1]);
        
        TextFileWriter w = new TextFileWriter();
        w.open("evaluationOutput.csv");
        for(int i = 0; i < nEvents*6; i++){
            String output = String.format("%f,%f,%f,%f", 
                   inputs[ 2].getDouble(new int[]{i,0}),
                   inputs[ 2].getDouble(new int[]{i,1}),
                    outputs[0].getDouble(new int[]{i,0}),
                    outputs[0].getDouble(new int[]{i,1})                    
                    );
            w.writeString(output);
        }
        w.close();
    }
    
    public void train(){
        /*
        BackpropagationTrainer trainer = network.getTrainer();
         trainer.setLearningRate(0.001f) // za ada delta 0.00001f za rms prop 0.001
                 //trainer.setLearningRate(0.01f) // za ada delta 0.00001f za rms prop 0.001
                 .setMaxError(0.0001f)
                 .setOptimizer(OptimizerType.SGD) // use adagrad optimization algorithm
                 .setL1Regularization(0.005f)
                 //.setL2Regularization(0.001f)
                 //.setBatchSize(200)
                 //.setBatchMode(true)
                 .setMaxEpochs(this.nEpochs);
        */
        INDArray[] inputs = getDummyInputs(1000);
        for(int i = 0; i < 1000; i++){
            long then = System.currentTimeMillis();
            network.fit(new INDArray[]{inputs[0],inputs[1]}, new INDArray[]{inputs[2]});
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i,network.score(), now-then);
        }
    }
    
    public void trainFile(String file, int nEvents){
       
        INDArray[] inputs = this.getFromFile(file, nEvents);        
        
        for(int i = 0; i < nEpochs; i++){
            long then = System.currentTimeMillis();
            network.fit(new INDArray[]{inputs[0],inputs[1]}, new INDArray[]{inputs[2]});
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i,network.score(), now-then);            
            if(i%25==0&&i!=0){
                this.save("level3_model_"+ this.cnnModel + "_" + i +"_epochs.network");
            }
        }        
        //network.output()
    }
    
    public INDArray[] getFromFile(String file, int max){
        HipoReader r = new HipoReader(file);
        CompositeNode nDC = new CompositeNode( 12, 1,  "bbsbil", 4096);
        CompositeNode nEC = new CompositeNode( 11, 2, "bbsbifs", 4096);
        CompositeNode nRC = new CompositeNode(  5, 1,  "b", 10);
        CompositeNode nET = new CompositeNode(  5, 2,  "b", 10);
        
        INDArray  DCArray = Nd4j.zeros( max*6 , 1, 6, 112);
    	INDArray  ECArray = Nd4j.zeros( max*6 , 1, 6,  72);
        INDArray OUTArray = Nd4j.zeros( max*6,  2);
        Event event = new Event();
        
        for(int i = 0; i < max; i++){
            r.nextEvent(event);
            event.read(nDC, 12, 1);
            event.read(nEC, 11, 2);
            event.read(nRC,  5, 1);
            //System.out.printf(" READ %d %d %d\n",nDC.getRows(),nEC.getRows(),nRC.getRows());
            Level3Utils.fillDC(DCArray, nDC, i);
            Level3Utils.fillDC(ECArray, nEC, i);
            Level3Utils.fillLabels(OUTArray, nRC, i);
        }
        
        return new INDArray[]{DCArray, ECArray, OUTArray};
    }
    
    public INDArray[]  getDummyInputs(int batch){
        INDArray  DCArray = Nd4j.zeros( batch*6 , 1, 6, 112);
    	INDArray  ECArray = Nd4j.zeros( batch*6 , 1, 6,  72);
        INDArray OUTArray = Nd4j.zeros( batch*6,  2);
        return new INDArray[]{DCArray, ECArray, OUTArray};
    }
    
    public static void main(String[] args){

        String file  = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.h5_000000_daq.h5";
        String file2 = "/Users/gavalian/Work/DataSpace/trigger/clas_005630.h5_000001_daq.h5";

        if(args.length>0) file = args[0];
        
        System.out.println(" training on file : " + file);
        Level3Trainer t = new Level3Trainer();
        
        /*t.cnnModel = "0b";
        t.initNetwork();
        t.nEpochs = 1250;
        t.trainFile(file, 10000);
        t.save("level3");
        */
        
        t.load("level3_model_0b_1225_epochs.network_0b.network");
        t.evaluateFile(file2, 2000);
        
    }
}
