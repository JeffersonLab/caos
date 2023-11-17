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
import org.nd4j.evaluation.classification.Evaluation;
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
public class Level3Trainer_Simulation{

    ComputationGraph network = null;
    public int nEpochs = 25;
    public String cnnModel = "0a";

    public Level3Trainer_Simulation() {

    }

    public void initNetwork(int nClasses) {
        ComputationGraphConfiguration config = Level3Models_MultiClass.getModel(cnnModel,nClasses);
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
            Logger.getLogger(Level3Trainer_Simulation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer_Simulation.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static int findIndexClassString(List<String[]> files,String className){
		int index=-1;
		for (int i=0;i<files.size();i++) {
			for(int j=0;j<files.get(i).length;j++){
				if (files.get(i)[j].contains(className)){index=i;}
			}
		}
		return index;
	}


    public void evaluateFile(List<String[]> files, int nEvents_pSample, Boolean doPlots) {

        MultiDataSet data = this.getClassesFromFile(files,nEvents_pSample,0.7);

        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);

        long nTestEvents = data.getFeatures()[0].shape()[0];

        int el_index=findIndexClassString(files,"el");

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics_MultiClass metrics = new Level3Metrics_MultiClass(nTestEvents, outputs[0], data.getLabels()[0],el_index,files.size(),doPlots);

    }

    public void trainFile(List<String[]> files, int nEvents_pSample, int nEvents_pSample_test,int batchSize) {

        MultiDataSet data = this.getClassesFromFile(files,nEvents_pSample,0);
        MultiDataSet data_test = this.getClassesFromFile(files,nEvents_pSample_test,0.7);

        Evaluation eval = new Evaluation(files.size());

        long NTotEvents = data.getFeatures()[0].shape()[0];
        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();

            long nBatches=NTotEvents/batchSize;
            for(int batch=0;batch<nBatches;batch++){
                int bS=batch*batchSize;
		        int bE=(batch+1)*batchSize;
                INDArray DC_b=data.getFeatures()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
		        INDArray EC_b=data.getFeatures()[1].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b=data.getLabels()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all());
                network.fit(new INDArray[] {DC_b,EC_b}, new INDArray[] {Lab_b});
            }

            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);
            INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1]);
            
		    eval.eval(data_test.getLabels()[0], outputs[0]);
            System.out.printf("Test Purity: %f, Efficiency: %f\n",eval.precision(),eval.recall());
            if (i % 50 == 0 && i != 0) {
                this.save("tmp_models/level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
        
    }

    public void getEnergiesForClassesFromFiles(List<String[]> files,List<String[]> names, int max) {
        //for testing
        TGCanvas c = new TGCanvas();
        c.setTitle("Calorimeter Energy");
        TGCanvas c2 = new TGCanvas();
        c2.setTitle("Number of Hits in Calorimeter");
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int added_classes=0, classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

            H1F h = new H1F("E", 101, -0.01, 1);
            h.attr().setLineColor(classs+2);// tags start at 1
            h.attr().setTitleX("Calorimeter Energy [AU]");
            h.attr().setLineWidth(3);
            h.attr().setTitle("Class " + Arrays.toString(names.get(classs)));
            H1F h2 = new H1F("E", 101, 0, 101);
            h2.attr().setLineColor(classs+2);// tags start at 1
            h2.attr().setTitleX("N Hits in Calorimeter");
            h2.attr().setLineWidth(3);
            h2.attr().setTitle("Class " + Arrays.toString(names.get(classs)));

            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                int nMax = max/file_arr.length;

                CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);
                    event.read(nEC, 11, 2);
                    Node node = event.read(5, 4);
                    int[] ids = node.getInt();

                    List<Double> energies = new ArrayList<Double>();
                    Level3Utils.fillEC(ECArray, nEC, ids[2], counter, energies);
                    for (double energy : energies) {
                        h.fill(energy);
                    }
                    h2.fill(energies.size());
                    counter++;

                }
            }
            if (classs == 0) {
                c.draw(h);
                c2.draw(h2);
            } else{
                c.draw(h,"same");
                c2.draw(h2,"same");
            }
            classs++;
        }
    }

    public void histClasses(List<String[]> files,List<String[]> names, int max) {
        TGCanvas c = new TGCanvas();
        c.setTitle("Tags");
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int added_classes=0, classs=0;

        for (String[] file_arr : files) {
            H1F h = new H1F("E", 80, 0, 80);
            h.attr().setLineColor(classs + 2);
            h.attr().setTitleX("Tags");
            h.attr().setLineWidth(3);
            h.attr().setTitle("Class " + Arrays.toString(names.get(classs)));

            System.out.printf("Class: %d",classs);

            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                int nMax = max/file_arr.length;

                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);

                    Node node = event.read(5, 4);
                    int[] ids = node.getInt();

                    // Event tag==4 is much more prevalent than others
                    // in the case where we have more than one tag per class
                    // probably don't want tag 4 to dominate

                    h.fill(event.getEventTag() * 10 + ids[1]);
                    counter++;

                }
            }
            if (classs == 0) {
                c.draw(h);
            } else {
                c.draw(h, "same");
            }
            classs++;
        }
    }

    public void saveClasses(List<String[]> files,List<String[]> names,String out, int max) {

        //added tag is for individual tag
        //classs is for each array of tags ie class
        int added_classes=0, classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                int nMax = max/file_arr.length;

                CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
                CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

                INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray OUTArray = Nd4j.zeros(nMax, files.size());
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

                    Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                    int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                    Level3Utils.fillLabels_MultiClass(OUTArray, files.size(), classs, counter);// tag
                    counter++;

                }
                System.out.printf("loaded samples (%d)\n\n\n", counter);
                File fileEC = new File(out + "EC_" + String.valueOf(names.get(classs)[j]) + ".npy");
                File fileDC = new File(out + "DC_" + String.valueOf(names.get(classs)[j]) + ".npy");
                try {
                    Nd4j.writeAsNumpy(ECArray, fileEC);
                    Nd4j.writeAsNumpy(DCArray, fileDC);
                } catch (IOException e) {
                    System.out.println("Could not write file");
                }
                added_classes++;
            }
            classs++;
        }

    }

    public MultiDataSet getClassesFromFile(List<String[]> files, int max,double trainTestP) {
        INDArray[] inputs = new INDArray[2];
        INDArray[] outputs = new INDArray[1];
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int added_classes=0, classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

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

                INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray OUTArray = Nd4j.zeros(nMax, files.size());
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
                        Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                        int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                        Level3Utils.fillLabels_MultiClass(OUTArray, files.size(), classs, counter);// tag
                        counter++;
                    }
                    eventNb++;

                }
                System.out.printf("loaded samples (%d)\n\n\n", counter);
                if (added_classes == 0) {
                    inputs = new INDArray[] { DCArray, ECArray };
                    outputs = new INDArray[] { OUTArray };
                } else {
                    inputs[0] = Nd4j.vstack(inputs[0], DCArray);
                    inputs[1] = Nd4j.vstack(inputs[1], ECArray);
                    outputs[0] = Nd4j.vstack(outputs[0], OUTArray);
                }
                added_classes++;

            }
            classs++;  

        }
        
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
        files.add(new String[] { dir+"el" });*/

        files.add(new String[] { dir+"pim", dir+"gamma" });
        files.add(new String[] { dir+"el" });

        List<String[]> names = new ArrayList<>();
        /*names.add(new String[] { "pim"});
        names.add(new String[] { "gamma"});
        names.add(new String[] { "el" });*/

        names.add(new String[] { "pim", "gamma" });
        names.add(new String[] { "el" });

        String net = "0d";
        Level3Trainer_Simulation t = new Level3Trainer_Simulation();

        /*t.getEnergiesForClassesFromFiles(files,names, 10000);
        t.histClasses(files,names, 10000);
        t.saveClasses(files,names, out, 50000);*/

        t.cnnModel = net;

        // if not transfer learning
        t.initNetwork(files.size());

        // transfer learning
        // t.load("level3_sim_"+net+".network");

        t.nEpochs = 500;//500
        t.trainFile(files,50000,10000,10000);//50000 10000 10000
        t.save("level3_sim");

        t.load("level3_sim_"+net+".network");
        t.evaluateFile(files,10000,false);//10000

    }
}
