/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.utils.io.OptionParser;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.datasets.iterator.loader.MultiDataSetLoaderIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
//import org.deeplearning4j.parallelism.ParallelInference;
//import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.nd4j.linalg.dataset.api.iterator.TestMultiDataSetIterator;

import twig.data.GraphErrors;
import twig.data.H1F;
import twig.data.H2F;
import twig.graphics.TGCanvas;
import twig.server.HttpDataServer;
import twig.server.HttpServerConfig;

/**
 *
 * @author gavalian, tyson
 */
public class Level3ClusterFinder_Simulation{

    ComputationGraph network = null;
    public int nEpochs = 25;
    public String cnnModel = "0a";

    public Level3ClusterFinder_Simulation() {

    }

    public void initNetwork() {
        ComputationGraphConfiguration config = Level3Models_ClusterFinder.getModel(cnnModel);
        network = new ComputationGraph(config);
        network.init();
        System.out.println(network.summary());
        //ParallelInference pi = new ParallelInference.Builder(network).workers(5).build();
    }


    public void testLayers(MultiDataSet data,int nEx){
        Layer lay=network.getLayer("L1");
        Layer lay2=network.getLayer("L2");
        /*Layer lay3=network.getLayer("L3");
        Layer lay4=network.getLayer("L4");*/

        ComputationGraphConfiguration configTest = Level3Models_ClusterFinder.getModel("testL1");
        ComputationGraph networkTest = new ComputationGraph(configTest);
        networkTest.init();
        networkTest.getLayer("L1").setParams(lay.params());
        INDArray[] out=networkTest.output(data.getFeatures()[0].get(NDArrayIndex.interval(0,nEx), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        
        ComputationGraphConfiguration configTest2 = Level3Models_ClusterFinder.getModel("testL2");
        ComputationGraph networkTest2 = new ComputationGraph(configTest2);
        networkTest2.init();
        networkTest2.getLayer("L1").setParams(lay.params());
         networkTest2.getLayer("L2").setParams(lay2.params());
        INDArray[] out2=networkTest2.output(data.getFeatures()[0].get(NDArrayIndex.interval(0,nEx), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        
        /*ComputationGraphConfiguration configTest4 = Level3Models_ClusterFinder.getModel("testL4");
        ComputationGraph networkTest4 = new ComputationGraph(configTest4);
        networkTest4.init();
        networkTest4.getLayer("L1").setParams(lay.params());
        networkTest4.getLayer("L2").setParams(lay2.params());
        networkTest4.getLayer("L3").setParams(lay3.params());
        networkTest4.getLayer("L4").setParams(lay4.params());
        INDArray[] out4=networkTest4.output(data.getFeatures()[0].get(NDArrayIndex.interval(0,nEx), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        */

        for(int i=0;i<nEx;i++){
            plotDCExamples(data.getFeatures()[0], i+1,i);
            plotDCExamples(out[0], i+1,i);
            plotDCExamples(out2[0], i+1,i);
            //plotDCExamples(out4[0], i+1,i);
        }
        
    }

    public void save(String file) {

        try {
            network.save(new File(file + "_" + cnnModel + ".network"));
            System.out.println("saved file : " + file + "\n" + "done...");
        } catch (IOException ex) {
            Logger.getLogger(Level3ClusterFinder_Simulation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3ClusterFinder_Simulation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void evaluateFile(List<String[]> files,List<String[]> names,String bg,List<Integer> nParts,List<Double> nStart, int nEvents_pSample, Boolean doPlots) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,nStart);

        if (nParts.size() > 0) {
            for (int nPart : nParts) {

                long nEvents_pSample_sp=nEvents_pSample;
                if (nPart != 0) {
                    long nEvents = data.getFeatures()[0].shape()[0];
                    while ((nEvents % nPart) != 0) {
                        nEvents--;
                    }
                    nEvents_pSample_sp = nEvents / nPart;
                }

                MultiDataSet data_nPart= makeSampleNPart(nPart, data,nEvents_pSample_sp);
                
                long nTestEvents = data_nPart.getFeatures()[0].shape()[0];

                if (bg != "") {
                    data_nPart = addBg(bg, (int) nTestEvents, 50, data_nPart);
                }

                //plotDCExamples(data_nPart.getFeatures()[0], 5,0);
                //plotFTOFExamples(data_nPart.getFeatures()[1], 10);

                //INDArray[] outputs = network.output(data_nPart.getFeatures()[0]);//0a,0b
                //INDArray[] outputs = network.output(data_nPart.getFeatures()[0], data_nPart.getFeatures()[1]);//0b,0c
                INDArray[] outputs = network.output(data_nPart.getFeatures()[0], data_nPart.getFeatures()[2]);//0e
                //INDArray[] outputs = network.output(data_nPart.getFeatures()[2]);//0f

                System.out.println("\n\nTesting with " + nPart + " particles (" + nTestEvents + " events)");
                Level3Metrics_ClusterFinder metrics = new Level3Metrics_ClusterFinder(nTestEvents, outputs[0],
                       data_nPart.getLabels()[0], doPlots);
            }
        }

        if(nParts.size()>0){
            data=makeMultiParticleSample(nParts,data);
        }

        long nTestEvents = data.getFeatures()[0].shape()[0];

        /*INDArray DC_out=data.getFeatures()[0].get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray Label_out=data.getLabels()[0].get(NDArrayIndex.point(0), NDArrayIndex.all());
        Nd4j.writeTxt(DC_out, "/Users/tyson/data_repo/trigger_data/sims/DC.csv");
        Nd4j.writeTxt(Label_out, "/Users/tyson/data_repo/trigger_data/sims/EC.csv");*/

        if(bg!=""){
            data=addBg(bg,(int) nTestEvents, 50, data);
        }

        //check if any elements are greater than 1
        /*int nElementGt1=0;
        for(int i=0;i<nTestEvents;i++){
            INDArray DCArray = data.getFeatures()[2].get(NDArrayIndex.point(i), NDArrayIndex.all(),
                    NDArrayIndex.all(),
                    NDArrayIndex.all());
            if(DCArray.gt(1).any()){
                nElementGt1++;
            }
        }
        System.out.printf("Have %d EC elements with entries greater than 1... \n",nElementGt1);*/

        //testLayers(data,1);
        //plotDCExamples(data.getFeatures()[0], 1,0);
        //plotFTOFExamples(data.getFeatures()[1], 10);
        //plotDCTDC(data.getFeatures()[0], (int)nTestEvents);
        //plotECINExamples(data.getFeatures()[2], 10);
            
        //INDArray[] outputs = network.output(data.getFeatures()[0]); //0a,0d
        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);//0b,0c
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[2]);//0e
        //INDArray[] outputs = network.output(data.getFeatures()[2]);//0f

        System.out.println("\n\nTesting with combined dataset ("+nTestEvents+" events)");
        Level3Metrics_ClusterFinder metrics = new Level3Metrics_ClusterFinder(nTestEvents, outputs[0], data.getLabels()[0],doPlots);

    }

    public void trainFile(List<String[]> files,List<String[]> names,String bg,List<Integer> nParts,List<Double> nStart,List<Double> nStart_t, int nEvents_pSample, int nEvents_pSample_test,int batchSize) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,nStart);
        
        MultiDataSet data_test = this.getClassesFromFile(files,names,nEvents_pSample_test,nStart_t);
        if(nParts.size()>0){
            data=makeMultiParticleSample(nParts,data);
            data_test=makeMultiParticleSample(nParts,data_test);
        }
        long NTotEvents = data.getFeatures()[0].shape()[0];
        long NTotEvents_test = data_test.getFeatures()[0].shape()[0];

        if(bg!=""){
            data=addBg(bg,(int) NTotEvents, 1, data);
            data_test=addBg(bg,(int) NTotEvents_test, 50, data_test);
        }

        RegressionEvaluation eval = new RegressionEvaluation(data.getLabels()[0].shape()[1]);

        
        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();

            long nBatches=NTotEvents/batchSize;
            for(int batch=0;batch<nBatches;batch++){
                int bS=batch*batchSize;
		        int bE=(batch+1)*batchSize;
                INDArray DC_b=data.getFeatures()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray FTOF_b=data.getFeatures()[1].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray ECIN_b=data.getFeatures()[2].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b=data.getLabels()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all());

                //network.fit(new INDArray[] {DC_b}, new INDArray[] {Lab_b});//0a,0d
                //network.fit(new INDArray[] {DC_b,FTOF_b}, new INDArray[] {Lab_b});//0b,0c
                network.fit(new INDArray[] {DC_b,ECIN_b}, new INDArray[] {Lab_b});//0e
                //network.fit(new INDArray[] {ECIN_b}, new INDArray[] {Lab_b});//0f
            }

            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);
            //INDArray[] outputs = network.output(data_test.getFeatures()[0]);//0a,0d
            //INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1]);//0b,0c
            INDArray[] outputs = network.output(data_test.getFeatures()[0],data_test.getFeatures()[2]);//0e
            //INDArray[] outputs = network.output(data_test.getFeatures()[2]);//0f
            
		    eval.eval(data_test.getLabels()[0], outputs[0]);
            System.out.printf("Test Average MAE: %f, MSE: %f\n",eval.averageMeanAbsoluteError(),eval.averageMeanSquaredError());
            if (i % 50 == 0 && i != 0) {
                this.save("tmp_models/level3CF_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
        
    }

    public static void plotDCExamples(INDArray DCall, int nExamples,int start){
        for (int k = start; k < nExamples; k++) {
            for (int l = 0; l < DCall.shape()[1]; l++) {
                TGCanvas c = new TGCanvas();
                c.setTitle("DC (Superlayer "+String.valueOf(l)+")");

                H2F hDC = new H2F("DC", (int) DCall.shape()[3], 0, (int) DCall.shape()[3], (int) DCall.shape()[2], 0,
                        (int) DCall.shape()[2]);
                hDC.attr().setTitleX("Wires");
                hDC.attr().setTitleY("Layers");
                hDC.attr().setTitle("DC");
                INDArray DCArray = DCall.get(NDArrayIndex.point(k), NDArrayIndex.point(l),
                        NDArrayIndex.all(),
                        NDArrayIndex.all());
                for (int i = 0; i < DCArray.shape()[0]; i++) {
                    for (int j = 0; j < DCArray.shape()[1]; j++) {
                        if (DCArray.getFloat(i, j) != 0) {
                            hDC.fill(j, i, DCArray.getFloat(i, j));
                        }
                    }
                }
                c.draw(hDC);
            }
        }

	}

    public static void plotDCTDC(INDArray DCall, int nExamples){
        TGCanvas c = new TGCanvas();
        c.setTitle("DC TDC");

        H1F hDC = new H1F("DC", 100, 0, 1);
        hDC.attr().setLineColor(2);
		hDC.attr().setLineWidth(3);
        hDC.attr().setTitleX("TDC");
        hDC.attr().setTitle("DC");
        double tdc_min=99999;
        double tdc_max=0;
        for (int k = 0; k < nExamples; k++) {
            
            for (int l = 0; l < DCall.shape()[1]; l++) {
                for (int i = 0; i < DCall.shape()[2]; i++) {
                    for (int j = 0; j < DCall.shape()[3]; j++) {
                        if (DCall.getFloat(k, l, i, j) != 0) {
                            hDC.fill(DCall.getFloat(k, l, i, j));
                            if (DCall.getFloat(k, l, i, j) < tdc_min) {
                                tdc_min = DCall.getFloat(k, l, i, j);
                            }
                            if (DCall.getFloat(k, l, i, j) > tdc_max) {
                                tdc_max = DCall.getFloat(k, l, i, j);
                            }
                        }
                    }
                }
            }
            
        }
        c.draw(hDC);
        System.out.printf("tdc min %f max %f\n\n",tdc_min,tdc_max);

    }

    public static void plotFTOFExamples(INDArray FTOFall, int nExamples){
        for (int k = 0; k < nExamples; k++) {
            TGCanvas c = new TGCanvas();
            c.setTitle("FTOF");

            H1F hFTOF = new H1F("FTOF", 62, 0, 62);
		    hFTOF.attr().setLineColor(2);
		    hFTOF.attr().setLineWidth(3);
		    hFTOF.attr().setTitleX("FTOF");
		    hFTOF.attr().setTitle("FTOF");
            INDArray FTOFArray = FTOFall.get(NDArrayIndex.point(k), NDArrayIndex.point(0),
                    NDArrayIndex.all(),
                    NDArrayIndex.point(0));
            for (int i = 0; i < 62; i++) {
                if (FTOFArray.getFloat(i) > 0) {
                    hFTOF.fill(i,FTOFArray.getFloat(i));
                }
            }
            c.draw(hFTOF);
        }

	}

    public static void plotECINExamples(INDArray ECINall, int nExamples){
        for (int k = 0; k < nExamples; k++) {
            TGCanvas c = new TGCanvas();
            c.setTitle("ECIN");

            H1F hECIN = new H1F("ECIN", 108, 0, 108);
		    hECIN.attr().setLineColor(2);
		    hECIN.attr().setLineWidth(3);
		    hECIN.attr().setTitleX("ECIN");
		    hECIN.attr().setTitle("ECIN");
            INDArray ECINArray = ECINall.get(NDArrayIndex.point(k), NDArrayIndex.point(0),
                    NDArrayIndex.point(0),NDArrayIndex.all());
            for (int i = 0; i < 108; i++) {
                if (ECINArray.getFloat(i) > 0) {
                    hECIN.fill(i,ECINArray.getFloat(i));
                }
            }
            c.draw(hECIN);
        }

	}

    public static void applyThreshold(INDArray predictions, double thresh){
        //DL4J doesn't have fancy boolean indexing :(
        for (int i=0;i<predictions.shape()[0];i++){
            for (int j=0;j<predictions.shape()[1];j++){
                if(predictions.getFloat(i, j)<thresh){
                    predictions.putScalar(new int[]{i,j}, 0);
                }
            }
        }
    }

    public MultiDataSet makeSampleNPart(int nPart,MultiDataSet dataset,long nEvents_pSample){
        //INDArray DC_out = Nd4j.zeros(1, 1, 36, 112);
        INDArray DC_out = Nd4j.zeros(1, 6, 6, 112);
        INDArray FTOF_out = Nd4j.zeros(1, 1,62,1);
        INDArray ECIN_out = Nd4j.zeros(1, 1,1,108);
        INDArray Label_out = Nd4j.zeros(1, 108);

        if (nPart != 0) {

            for (int i = 0; i < nPart; i++) {
                long bS = i * nEvents_pSample;
                long bE = (i + 1) * nEvents_pSample;
                INDArray DC_b = dataset.getFeatures()[0].get(NDArrayIndex.interval(bS, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray FTOF_b = dataset.getFeatures()[1].get(NDArrayIndex.interval(bS, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray ECIN_b = dataset.getFeatures()[2].get(NDArrayIndex.interval(bS, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b = dataset.getLabels()[0].get(NDArrayIndex.interval(bS, bE), NDArrayIndex.all());
                if (i == 0) {
                    DC_out = DC_b;
                    FTOF_out=FTOF_b;
                    ECIN_out=ECIN_b;
                    Label_out = Lab_b;
                } else {
                    DC_out = addInputArrays(DC_out, DC_b);
                    FTOF_out = addInputArrays(FTOF_out, FTOF_b);
                    ECIN_out = addInputArrays(ECIN_out, ECIN_b);
                    Label_out = addLabelArrays(Label_out, Lab_b);
                }
            }
        } else {
            //DC_out = Nd4j.zeros(nEvents_pSample, 1, 36, 112);
            DC_out = Nd4j.zeros(nEvents_pSample, 6, 6, 112);
            FTOF_out = Nd4j.zeros(nEvents_pSample, 1, 62, 1);
            ECIN_out = Nd4j.zeros(nEvents_pSample, 1, 1, 108);
            Label_out = Nd4j.zeros(nEvents_pSample, 108);
        }

        MultiDataSet dataset_out = new MultiDataSet(new INDArray[]{DC_out,FTOF_out,ECIN_out},new INDArray[]{Label_out});
        return dataset_out;
    }

    public MultiDataSet makeMultiParticleSample(List<Integer> nParts, MultiDataSet dataset) {

        INDArray[] inputs = new INDArray[3];
        INDArray[] outputs = new INDArray[1];
        int added=0,addedNonZero=1;

        //split dataset into data with 1 track
        //data with 2 tracks being combined
        //data with 3 etc
        int sizeNParts=nParts.size()+1;
        //not combining data when requiring 0 particles
        if(nParts.contains(0)){
            sizeNParts--;
        }
        long totLength=dataset.getFeatures()[0].shape()[0]/sizeNParts;

        //find minimum amount of data in one sample
        long minEvents_pSample=99999;
        for (int nPart : nParts) {
            if (nPart != 0) {
                long nEvents = dataset.getFeatures()[0].shape()[0]/sizeNParts;
                while ((nEvents % nPart) != 0) {
                    nEvents--;
                }
                long nEvents_pSample = nEvents / nPart;
                if (nEvents_pSample < minEvents_pSample) {
                    minEvents_pSample = nEvents_pSample;
                }
            }
        }

        for (int nPart : nParts) {
            INDArray[] inputs_4part = new INDArray[3];
            INDArray[] outputs_4part = new INDArray[1];
            if (nPart != 0) {
                //addedNonZero starts at 1 as single particle sample is added by default
                inputs_4part[0] = dataset.getFeatures()[0].get(
                        NDArrayIndex.interval((addedNonZero) * totLength, (addedNonZero + 1) * totLength),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                inputs_4part[1] = dataset.getFeatures()[1].get(
                        NDArrayIndex.interval((addedNonZero) * totLength, (addedNonZero + 1) * totLength),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                inputs_4part[2] = dataset.getFeatures()[2].get(
                        NDArrayIndex.interval((addedNonZero) * totLength, (addedNonZero + 1) * totLength),
                        NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                outputs_4part[0] = dataset.getLabels()[0].get(
                        NDArrayIndex.interval((addedNonZero) * totLength, (addedNonZero + 1) * totLength),
                        NDArrayIndex.all());
                addedNonZero++;
            }
            MultiDataSet dataset_4part=new MultiDataSet(inputs_4part,outputs_4part);

            MultiDataSet data_nPart = makeSampleNPart(nPart, dataset_4part,minEvents_pSample);
            if(added==0){
                inputs[0] = Nd4j.vstack(
                        dataset.getFeatures()[0].get(NDArrayIndex.interval(0, minEvents_pSample), NDArrayIndex.all(),
                                NDArrayIndex.all(), NDArrayIndex.all()),
                        data_nPart.getFeatures()[0]);
                inputs[1] = Nd4j.vstack(
                        dataset.getFeatures()[1].get(NDArrayIndex.interval(0, minEvents_pSample), NDArrayIndex.all(),
                                NDArrayIndex.all(), NDArrayIndex.all()),
                        data_nPart.getFeatures()[1]);
                inputs[2] = Nd4j.vstack(
                        dataset.getFeatures()[2].get(NDArrayIndex.interval(0, minEvents_pSample), NDArrayIndex.all(),
                                NDArrayIndex.all(), NDArrayIndex.all()),
                        data_nPart.getFeatures()[2]);
                outputs[0] = Nd4j.vstack(
                        dataset.getLabels()[0].get(NDArrayIndex.interval(0, minEvents_pSample), NDArrayIndex.all()),
                        data_nPart.getLabels()[0]);
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], data_nPart.getFeatures()[0]);
                inputs[1] = Nd4j.vstack(inputs[1], data_nPart.getFeatures()[1]);
                inputs[2] = Nd4j.vstack(inputs[2], data_nPart.getFeatures()[2]);
                outputs[0] = Nd4j.vstack(outputs[0], data_nPart.getLabels()[0]);
            }
            added++;
        }
        MultiDataSet dataset_out = new MultiDataSet(inputs, outputs);
        dataset_out.shuffle();
        return dataset_out;
    }

    public INDArray addInputArrays(INDArray arr1, INDArray arr2){
        if (arr1.equalShapes(arr2)) {
            for (int i = 0; i < arr1.shape()[0]; i++) {
                for (int k = 0; k < arr1.shape()[1]; k++) {
                    for (int l = 0; l < arr1.shape()[2]; l++) {
                        for (int m = 0; m < arr1.shape()[3]; m++) {
                            //if no entry in array 1 and there's an entry in array 2, then add it to array 1
                            if(arr1.getFloat(i,k,l, m) == 0 && arr2.getFloat(i,k,l, m) != 0){
                                arr1.putScalar(new int[]{i,k,l, m}, arr2.getFloat(i,k,l, m));
                            }
                            //if there is an entry in array 1, we keep it
                            //never add array 2 to array 1 if array 1 already has an entry
                        }
                    }
                }
            }
        }else{
            System.out.println("****** Array shapes don't match, returning first array ******");
        }
        return arr1;

    }

    public INDArray addLabelArrays(INDArray arr1, INDArray arr2){
        if (arr1.equalShapes(arr2)) {
            for (int i = 0; i < arr1.shape()[0]; i++) {
                for (int k = 0; k < arr1.shape()[1]; k++) {
                    // if no entry in array 1 and there's an entry in array 2, then add it to array 1
                    
                    if (arr1.getFloat(i, k) == 0 && arr2.getFloat(i, k) != 0) {
                        arr1.putScalar(new int[] { i, k}, arr2.getFloat(i, k));
                    }
                    // if there is an entry in array 1, we keep it
                    // never add array 2 to array 1 if array 1 already has an entry
                }
            }
        }else{
            System.out.println("****** Array shapes don't match, returning first array ******");
        }
        return arr1;

    }

    public MultiDataSet addBg(String bgLoc, int max,int start,MultiDataSet dataset) {
       
        int added = 0;
        //INDArray DCArray = Nd4j.zeros(max, 1, 36, 112);
        INDArray DCArray = Nd4j.zeros(max, 6, 6, 112);
        INDArray FTOFArray = Nd4j.zeros(max, 1,62,1);
        INDArray ECINArray = Nd4j.zeros(max, 1,1,108);
        while (added < max && start<101) {
            String file = bgLoc + "daq_"+String.valueOf(start)+".h5";
            start++;
            HipoReader r = new HipoReader();

            r.open(file);

            System.out.println("Reading file: " + file);

            int nMax = max;

            if (r.entries() < nMax)
                nMax = r.entries();

            CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
            CompositeNode nFTOF = new CompositeNode( 13, 3,  "bbsbifs", 4096);
            CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

            Event event = new Event();
            int counter = 0;
            while (r.hasNext() == true && counter < nMax) {
                r.nextEvent(event);

                event.read(nDC, 12, 1);
                event.read(nFTOF, 13, 3);
                event.read(nEC, 11, 2);

                Node node = event.read(5, 4);

                int[] ids = node.getInt();

                //Level3Utils.fillDC_wLayers(DCArray, nDC, ids[2], counter);//_wLayersTDC
                Level3Utils.fillDC_SepSL(DCArray, nDC, ids[2], counter);
                Level3Utils.fillFTOF_wNorm(FTOFArray,nFTOF,ids[2],counter);
                Level3Utils.fillECin(ECINArray,nEC,ids[2],counter);

                counter++;
                added++;
            }
        }

        INDArray[] inputs = new INDArray[3];//1 with only DC
        INDArray[] outputs = new INDArray[1];
        inputs[0]=addInputArrays(dataset.getFeatures()[0],DCArray);
        inputs[1]=addInputArrays(dataset.getFeatures()[1],FTOFArray);
        inputs[2]=addInputArrays(dataset.getFeatures()[2],ECINArray);
        outputs[0]=dataset.getLabels()[0];
        dataset = new MultiDataSet(inputs,outputs);
        return dataset;
    }

    public MultiDataSet getClassesFromFile(List<String[]> files,List<String[]> names, int max,List<Double> nStart) {
        INDArray[] inputs = new INDArray[3];//1 with only DC
        INDArray[] outputs = new INDArray[1];
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

            INDArray[] inputs_class = new INDArray[3];//1 with only DC
            INDArray[] outputs_class = new INDArray[1];
            int added_classes=0;
            for (int j = 0; j < file_arr.length; j++) {
                
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                double trainTestP=nStart.get(classs);
                int start=(int)Math.ceil(trainTestP*r.entries());

                int nMax = max/file_arr.length;

                if (r.entries()< (nMax+start))
                    nMax = (r.entries()-start);

                CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
                CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
                CompositeNode nFTOF = new CompositeNode( 13, 3,  "bbsbifs", 4096);

                //INDArray DCArray = Nd4j.zeros(nMax, 1, 36, 112);
                INDArray DCArray = Nd4j.zeros(nMax, 6, 6, 112);
                INDArray FTOFArray = Nd4j.zeros(nMax, 1,62,1);
                INDArray ECINArray = Nd4j.zeros(nMax, 1,1,108);
                INDArray OUTArray = Nd4j.zeros(nMax, 108);
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);

                    event.read(nDC, 12, 1);
                    event.read(nEC, 11, 2);
                    event.read(nFTOF, 13, 3);

                    Node node = event.read(5, 4);

                    int[] ids = node.getInt();

                    // System.out.printf("event tag (%d) & ID (%d)\n", ids[1], ids[0]);
                    // System.out.printf("event tag (%d) & ID (%d)\n",event.getEventTag(),ids[0]);

                    //allows us to keep N last events for testing
                    if (eventNb >= start ) { //&& ids[1]==5tag 5 is 4.5 - 5.5 GeV
                        
                        Level3Utils.fillLabels_ClusterFinder(OUTArray, nEC,ids[2], counter);
                        INDArray EventOUTArray = OUTArray.get(NDArrayIndex.point(counter),NDArrayIndex.all());
                        if (EventOUTArray.any()) { 
                            //Level3Utils.fillDC_wLayers(DCArray, nDC, ids[2], counter);//_wLayersTDC
                            Level3Utils.fillDC_SepSL(DCArray, nDC, ids[2], counter);
                            Level3Utils.fillFTOF_wNorm(FTOFArray,nFTOF,ids[2],counter);
                            Level3Utils.fillECin(ECINArray,nEC,ids[2],counter);
                            counter++;
                        } else{
                            OUTArray.get(NDArrayIndex.point(counter), NDArrayIndex.all()).assign(Nd4j.zeros(1, 108));
                        }
                        
                    }
                    eventNb++;

                }

                System.out.printf("loaded samples (%d)\n\n\n", counter);
                if (added_classes == 0) {
                    //inputs = new INDArray[] { DCArray};
                    inputs_class = new INDArray[] { DCArray,FTOFArray,ECINArray};
                    outputs_class = new INDArray[] { OUTArray };
                } else {
                    inputs_class[0] = Nd4j.vstack(inputs_class[0], DCArray);
                    inputs_class[1] = Nd4j.vstack(inputs_class[1], FTOFArray);
                    inputs_class[2] = Nd4j.vstack(inputs_class[2], ECINArray);
                    outputs_class[0] = Nd4j.vstack(outputs_class[0], OUTArray);
                }
                added_classes++;

            }

            if (names.get(classs)[0] == "mixMatch") {
                System.out.println("mix matching");
                // Shuffle DC and FTOF arrays independently
                // Creates Calorimeter hits uncorrelated to DC tracks, FTOF hits
                MultiDataSet datasetDC = new MultiDataSet(new INDArray[]{inputs_class[0]},new INDArray[]{inputs_class[0]});
                datasetDC.shuffle();
                inputs_class[0] = datasetDC.getFeatures()[0];
                MultiDataSet datasetFTOF = new MultiDataSet(new INDArray[]{inputs_class[1]},new INDArray[]{inputs_class[1]});
                datasetFTOF.shuffle();
                inputs_class[1] = datasetFTOF.getFeatures()[0];
                /*MultiDataSet datasetECIN = new MultiDataSet(new INDArray[]{inputs_class[2]},new INDArray[]{outputs_class[0]});
                datasetECIN.shuffle();
                inputs_class[2] = datasetECIN.getFeatures()[0];
                outputs_class[0]=datasetECIN.getLabels()[0]; */
                // Note: OUTArray is EC, this should now be shuffled compared to DC,FTOF
                // If we use ECIN as input we need to shuffle ECIN input and output in same way.
                // This has same effect as not shuffling either.
            }

            //create sample where we have one DC track but 2 ECIN clusters
            //best used with electrons
            //divides nb events by 2, assumes even nb of events
            else if(names.get(classs)[0]=="1t2c"){
                long nEv=inputs_class[0].shape()[0];
                long bE = nEv/2;
                INDArray DC_a = inputs_class[0].get(NDArrayIndex.interval(0, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray FTOF_a = inputs_class[1].get(NDArrayIndex.interval(0, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray ECIN_a = inputs_class[2].get(NDArrayIndex.interval(0, bE), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_a = outputs_class[0].get(NDArrayIndex.interval(0, bE), NDArrayIndex.all());
                INDArray ECIN_b = inputs_class[2].get(NDArrayIndex.interval(bE,nEv), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b = outputs_class[0].get(NDArrayIndex.interval(bE,nEv), NDArrayIndex.all());
                inputs_class[0]=DC_a;
                inputs_class[1]=FTOF_a;
                inputs_class[2]=addInputArrays(ECIN_a,ECIN_b);
                outputs_class[0]=addLabelArrays(Lab_a, Lab_b);
            }

            //create sample where 2 DC superlayers are removed
            //ECIN input unchanged
            //ECIN output set to 0
            //aim to force network to use all six superlayers
            else if(names.get(classs)[0]=="corrupt1"){
                long nEv=inputs_class[0].shape()[0];
                Random rand = new Random();
                for(int i=0;i<nEv;i++){
                    //SLs to skip
                    int SLs1 = rand.nextInt(6);
                    int SLs2 = SLs1;
                    while(SLs1==SLs2){SLs2=rand.nextInt(6);}
                    inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs1), NDArrayIndex.all(),
                                            NDArrayIndex.all()).assign(Nd4j.zeros(6, 112));
                    inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs2), NDArrayIndex.all(),
                                            NDArrayIndex.all()).assign(Nd4j.zeros(6, 112));
                }
                outputs_class[0]=Nd4j.zeros(nEv, 108);
            }

            //create sample where 2 DC superlayers are shifted
            //ECIN input unchanged
            //ECIN output set to 0
            //aim to force network to use all six superlayers
            else if(names.get(classs)[0]=="corrupt2"){
                long nEv=inputs_class[0].shape()[0];
                Random rand = new Random();
                for(int i=0;i<nEv;i++){
                    //SLs to corrupt
                    int SLs1 = rand.nextInt(6);
                    int SLs2 = SLs1;
                    while(SLs1==SLs2){SLs2=rand.nextInt(6);}
                    int change1=rand.nextInt(112);  
                    int change2=rand.nextInt(112);
                    for (int l = 0; l < inputs_class[0].shape()[2]; l++) {
                        for (int m = 0; m < inputs_class[0].shape()[3]; m++) {
                            if(inputs_class[0].getFloat(i,SLs1,l, m) != 0){
                                //System.out.printf("SL %d, ev %d, m %d, l %d, WS %d, (111-m)-4+1 %d, val %f \n",SLs1,i,m,l,Ws1,(111-m)-4+1,inputs_class[0].getFloat(i,SLs1,l, m));
                                inputs_class[0].putScalar(new int[]{i,SLs1,l, change1}, inputs_class[0].getFloat(i,SLs1,l, m));
                                inputs_class[0].putScalar(new int[]{i,SLs1,l, m}, 0);
                            }
                            if(inputs_class[0].getFloat(i,SLs2,l, m) != 0){
                                //System.out.printf("SL %d, ev %d, m %d, l %d, WS %d, (111-m)-4+1 %d, val %f \n",SLs2,i,m,l,Ws2,(111-m)-4+1,inputs_class[0].getFloat(i,SLs2,l, m));
                                inputs_class[0].putScalar(new int[]{i,SLs2,l, change2}, inputs_class[0].getFloat(i,SLs2,l, m));
                                inputs_class[0].putScalar(new int[]{i,SLs2,l, m}, 0);
                            }
                        }
                    }
                }
                outputs_class[0]=Nd4j.zeros(nEv, 108);
            }

            if (classs == 0) {
                // inputs = new INDArray[] { DCArray, ECArray };
                inputs = inputs_class;
                outputs = outputs_class;
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], inputs_class[0]);
                inputs[1] = Nd4j.vstack(inputs[1], inputs_class[1]);
                inputs[2] = Nd4j.vstack(inputs[2], inputs_class[2]);
                outputs[0] = Nd4j.vstack(outputs[0], outputs_class[0]);
            }
            classs++;  

        }
       
        System.out.println(outputs[0]);

        MultiDataSet dataset = new MultiDataSet(inputs,outputs);
        dataset.shuffle();
        return dataset;
    }

    public static void main(String[] args) {
     
        String dir = "/Users/tyson/data_repo/trigger_data/sims/";

        //String dir = "/scratch/clasrun/caos/sims/";

        String bg="";//dir+"bg_50nA_10p6/";//"";

        List<String[]> files = new ArrayList<>();
        //files.add(new String[] { dir+"pim"});
        /*files.add(new String[] { dir+"gamma"});
        files.add(new String[] { dir+"pos"});
        files.add(new String[] {dir+"pim",dir+"pos",dir+"el",dir+"gamma"});*/
        files.add(new String[] { dir+"el" });
        files.add(new String[] { dir+"el" });
        //files.add(new String[] { dir+"pos"});

        List<String[]> names = new ArrayList<>();
        //names.add(new String[] { "pim"});
        /*names.add(new String[] { "gamma"});
        names.add(new String[] { "pos" });
        names.add(new String[]{"mixMatch","mixMatch","mixMatch","mixMatch"});*/
        //names.add(new String[] { "mixMatch" });
        //names.add(new String[] { "1t2c" });
        names.add(new String[] { "corrupt1" });
        names.add(new String[] { "el" });
        //names.add(new String[] { "pos" });

        List<Double> nStart=new ArrayList<>();
        nStart.add(0.0);
        nStart.add(0.5);

        List<Double> nStart_t=new ArrayList<>();
        nStart_t.add(0.9);
        nStart_t.add(0.95);

        //assumes at least one particle by default
        List<Integer> nParts=new ArrayList<>();
        //nParts.add(3);
        //nParts.add(4);
        //nParts.add(2);
        //nParts.add(0);


        String net = "0e";
        Level3ClusterFinder_Simulation t = new Level3ClusterFinder_Simulation();

        t.cnnModel = net;
        // if not transfer learning
        t.initNetwork();

        // transfer learning
        // t.load("level3CF_sim_"+net+".network");

        /*t.nEpochs = 750;//500
        t.trainFile(files,names,bg,nParts,nStart,nStart_t,70000,1000,1000);//30000 5000 10000
        t.save("level3CF_sim");*/

        t.load("level3CF_sim_"+net+"_noise2Tracks2Ch.network"); //_noise2Tracks //_noise2Tracks2Ch
        t.evaluateFile(files,names,bg,nParts,nStart_t,5000,true);//5000

    }
}
