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

    public void evaluateFile(List<String[]> files,List<String[]> names, int nEvents_pSample, Boolean doPlots) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,0.8);

        INDArray[] outputs = network.output(data.getFeatures()[0]);

        long nTestEvents = data.getFeatures()[0].shape()[0];

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics_ClusterFinder metrics = new Level3Metrics_ClusterFinder(nTestEvents, outputs[0], data.getLabels()[0],doPlots);

    }

    public void trainFile(List<String[]> files,List<String[]> names, int nEvents_pSample, int nEvents_pSample_test,int batchSize) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,0);
        MultiDataSet data_test = this.getClassesFromFile(files,names,nEvents_pSample_test,0.8);

        RegressionEvaluation eval = new RegressionEvaluation(data.getLabels()[0].shape()[1]);

        long NTotEvents = data.getFeatures()[0].shape()[0];
        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();

            long nBatches=NTotEvents/batchSize;
            for(int batch=0;batch<nBatches;batch++){
                int bS=batch*batchSize;
		        int bE=(batch+1)*batchSize;
                INDArray DC_b=data.getFeatures()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b=data.getLabels()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all());

                network.fit(new INDArray[] {DC_b}, new INDArray[] {Lab_b});
            }

            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);
            //INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1]);
            INDArray[] outputs = network.output(data_test.getFeatures()[0]);
            
		    eval.eval(data_test.getLabels()[0], outputs[0]);
            System.out.printf("Test Average MAE: %f, MSE: %f\n",eval.averageMeanAbsoluteError(),eval.averageMeanSquaredError());
            if (i % 50 == 0 && i != 0) {
                this.save("tmp_models/level3CF_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
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

    public MultiDataSet getClassesFromFile(List<String[]> files,List<String[]> names, int max,double trainTestP) {
        INDArray[] inputs = new INDArray[1];
        INDArray[] outputs = new INDArray[1];
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

            INDArray[] inputs_class = new INDArray[1];
            INDArray[] outputs_class = new INDArray[1];
            int added_classes=0;
            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                int start=(int)Math.ceil(trainTestP*r.entries());

                int nMax = max/file_arr.length;

                if (r.entries()< (nMax+start))
                    nMax = (r.entries()-start);

                CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
                CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

                INDArray DCArray = Nd4j.zeros(nMax, 1, 36, 112);
                INDArray OUTArray = Nd4j.zeros(nMax, 108);
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);

                    event.read(nDC, 12, 1);
                    event.read(nEC, 11, 2);

                    Node node = event.read(5, 4);

                    int[] ids = node.getInt();

                    // System.out.printf("event tag (%d) & ID (%d)\n", ids[1], ids[0]);
                    // System.out.printf("event tag (%d) & ID (%d)\n",event.getEventTag(),ids[0]);

                    //allows us to keep N last events for testing
                    if (eventNb >= start) {
                        
                        Level3Utils.fillLabels_ClusterFinder(OUTArray, nEC,ids[2], counter);
                        INDArray EventOUTArray = OUTArray.get(NDArrayIndex.point(counter),NDArrayIndex.all());
                        if (EventOUTArray.any()) { 
                            Level3Utils.fillDC_wLayers(DCArray, nDC, ids[2], counter);
                            counter++;
                        } else{
                            DCArray.get(NDArrayIndex.point(counter), NDArrayIndex.all()).assign(Nd4j.zeros(1, 108));
                        }
                        
                    }
                    eventNb++;

                }

                System.out.printf("loaded samples (%d)\n\n\n", counter);
                if (added_classes == 0) {
                    //inputs = new INDArray[] { DCArray, ECArray };
                    inputs_class = new INDArray[] { DCArray};
                    outputs_class = new INDArray[] { OUTArray };
                } else {
                    inputs_class[0] = Nd4j.vstack(inputs_class[0], DCArray);
                    outputs_class[0] = Nd4j.vstack(outputs_class[0], OUTArray);
                }
                added_classes++;

            }

            if (names.get(classs)[0] == "mixMatch") {
                System.out.println("mix matching");
                // Shuffle DC and EC arrays independently
                // Creates Calorimeter hits uncorrelated to DC tracks
                MultiDataSet datasetDC = new MultiDataSet(new INDArray[] { inputs_class[0] },
                        new INDArray[] { inputs_class[0] });
                datasetDC.shuffle();
                inputs_class[0] = datasetDC.getFeatures()[0];
                // Note: OUTArray is EC, this should now be shuffled compared to DC
            }

            if (classs == 0) {
                // inputs = new INDArray[] { DCArray, ECArray };
                inputs = inputs_class;
                outputs = outputs_class;
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], inputs_class[0]);
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
     
        //String dir = "/Users/tyson/data_repo/trigger_data/sims/";
        //String out = "/Users/tyson/data_repo/trigger_data/sims/python/";

        String dir = "/scratch/clasrun/caos/sims/";

        List<String[]> files = new ArrayList<>();
        /*files.add(new String[] { dir+"pim"});
        files.add(new String[] { dir+"gamma"});
        files.add(new String[] { dir+"pos"});
        files.add(new String[] {dir+"pim",dir+"pos",dir+"el",dir+"gamma"});*/
        files.add(new String[] { dir+"el" });

        List<String[]> names = new ArrayList<>();
        /*names.add(new String[] { "pim"});
        names.add(new String[] { "gamma"});
        names.add(new String[] { "pos" });
        names.add(new String[]{"mixMatch","mixMatch","mixMatch","mixMatch"});*/
        names.add(new String[] { "el" });

        String net = "0a";
        Level3ClusterFinder_Simulation t = new Level3ClusterFinder_Simulation();

        t.cnnModel = net;
        // if not transfer learning
        t.initNetwork();

        // transfer learning
        // t.load("level3CF_sim_"+net+".network");

        t.nEpochs = 750;//500
        t.trainFile(files,names,30000,1000,1000);//30000 5000 10000
        t.save("level3CF_sim");

        t.load("level3CF_sim_"+net+".network");
        t.evaluateFile(files,names,1000,false);//5000

    }
}
