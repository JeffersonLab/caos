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
import twig.data.H2F;
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


    public void evaluateFile(List<String[]> files,List<String[]> names, int nEvents_pSample, Boolean doPlots) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,0.8);

        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[2], data.getFeatures()[3]);

        long nTestEvents = data.getFeatures()[0].shape()[0];

        int el_index=findIndexClassString(files,"el");

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics_MultiClass metrics = new Level3Metrics_MultiClass(nTestEvents, outputs[0], data.getLabels()[0],el_index,files.size(),doPlots);

    }

    public void plotHTCC(List<String[]> files,List<String[]> names, int max) {

        H1F hHTCC_all = new H1F("HTCC_all", 8, 1, 9);
        hHTCC_all.attr().setTitleX("Mirror");
        hHTCC_all.attr().setTitleY("ADC [AU]");
        hHTCC_all.attr().setTitle("HTCC");
        hHTCC_all.attr().setLineColor(2);
		hHTCC_all.attr().setLineWidth(3);

        H2F hHTCC2D_all = new H2F("HTCC2D_all", 4, 1, 5, 2, 1, 3);
        hHTCC2D_all.attr().setTitleX("Mirror");
        hHTCC2D_all.attr().setTitleY("Layers");
        hHTCC2D_all.attr().setTitle("HTCC");

        MultiDataSet data = this.getClassesFromFile(files,names,max,0);
        INDArray HTCC=data.getFeatures()[3];
        for (int k=0;k<max;k++){

            H1F hHTCC_adc = new H1F("HTCC_adc", 100, 0, 1);
            hHTCC_adc.attr().setTitleX("ADC Single Event [AU]");
            hHTCC_adc.attr().setTitle("HTCC ADC Single Event");
            hHTCC_adc.attr().setLineColor(2);
		    hHTCC_adc.attr().setLineWidth(3);

            H1F hHTCC = new H1F("HTCC", 8, 1, 9);
            hHTCC.attr().setTitleX("Mirror");
            hHTCC.attr().setTitleY("ADC [AU]");
            hHTCC.attr().setTitle("HTCC");
            hHTCC.attr().setLineColor(2);
		    hHTCC.attr().setLineWidth(3);
            

            H2F hHTCC2D = new H2F("HTCC2D", 4, 1, 5, 2, 1, 3);
            hHTCC2D.attr().setTitleX("Mirror");
            hHTCC2D.attr().setTitleY("Layers");
            hHTCC2D.attr().setTitle("HTCC");
            INDArray HTCCArray = HTCC.get(NDArrayIndex.point(k), NDArrayIndex.point(0),
                    NDArrayIndex.all(),
                     NDArrayIndex.point(0));
            for (int i = 0; i < 8; i++) {
                int j=1;
                if(i>3){j=2;}
                if (HTCCArray.getFloat(i) > 0) {
                    double mirror=(i)%4;
                    hHTCC2D.fill(mirror+1,j,HTCCArray.getFloat(i));
                    hHTCC2D_all.fill(mirror+1,j,HTCCArray.getFloat(i));
                    hHTCC.fill(i+1,HTCCArray.getFloat(i));
                    hHTCC_all.fill(i+1,HTCCArray.getFloat(i));
                    hHTCC_adc.fill(HTCCArray.getFloat(i));
                }
            }
            if(k<10){
                TGCanvas c2D = new TGCanvas();
                c2D.setTitle("HTCC");
                c2D.draw(hHTCC2D);

                TGCanvas c = new TGCanvas();
                c.setTitle("HTCC");
                c.draw(hHTCC);

                /*TGCanvas c_adc = new TGCanvas();
                c_adc.setTitle("HTCC ADC");
                c_adc.draw(hHTCC_adc);*/
            }

        }

        TGCanvas c2D_all = new TGCanvas();
        c2D_all.setTitle("HTCC");
        c2D_all.draw(hHTCC2D_all);

        TGCanvas c_all = new TGCanvas();
        c_all.setTitle("HTCC");
        c_all.draw(hHTCC_all);

        for(int l=0;l<8;l++){

            H1F hHTCC_adc = new H1F("HTCC_adc", 100, 0, 1);
            hHTCC_adc.attr().setTitleX("ADC Mirror "+String.valueOf(l+1)+" [AU]");
            hHTCC_adc.attr().setTitle("HTCC ADC Mirror "+String.valueOf(l+1));
            hHTCC_adc.attr().setLineColor(2);
		    hHTCC_adc.attr().setLineWidth(3);
            for (int k = 0; k < max; k++) {

                INDArray HTCCArray = HTCC.get(NDArrayIndex.point(k), NDArrayIndex.point(0),
                        NDArrayIndex.all(),
                        NDArrayIndex.point(0));
                hHTCC_adc.fill(HTCCArray.getFloat(l));
                    
            }
            TGCanvas c_adc = new TGCanvas();
            c_adc.setTitle("HTCC");
            c_adc.draw(hHTCC_adc);
        }

    }

    public void trainFile(List<String[]> files,List<String[]> names, int nEvents_pSample, int nEvents_pSample_test,int batchSize) {

        MultiDataSet data = this.getClassesFromFile(files,names,nEvents_pSample,0);
        MultiDataSet data_test = this.getClassesFromFile(files,names,nEvents_pSample_test,0.8);

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
                INDArray FTOF_b=data.getFeatures()[2].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
		        INDArray HTCC_b=data.getFeatures()[3].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                INDArray Lab_b=data.getLabels()[0].get(NDArrayIndex.interval(bS,bE), NDArrayIndex.all());

                //System.out.printf("1 %d, 2 %d, 3 %d, 4%d\n",FTOF_b.shape()[0],FTOF_b.shape()[1],FTOF_b.shape()[2],FTOF_b.shape()[3]);

                //network.fit(new INDArray[] {DC_b,EC_b}, new INDArray[] {Lab_b});
                network.fit(new INDArray[] {DC_b,EC_b,FTOF_b,HTCC_b}, new INDArray[] {Lab_b});
            }

            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);
            //INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1]);
            INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1], data_test.getFeatures()[2], data_test.getFeatures()[3]);
            
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
        TGCanvas c_FTOF = new TGCanvas();
        c_FTOF.setTitle("FTOF ADC");
        TGCanvas c2_FTOF = new TGCanvas();
        c2_FTOF.setTitle("Number of Hits in FTOF");
        TGCanvas c_HTCC = new TGCanvas();
        c_HTCC.setTitle("HTCC ADC");
        TGCanvas c2_HTCC = new TGCanvas();
        c2_HTCC.setTitle("Number of Hits in HTCC");
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

            H1F h_FTOF = new H1F("E_FTOF", 101, -0.01, 1);
            h_FTOF.attr().setLineColor(classs+2);// tags start at 1
            h_FTOF.attr().setTitleX("FTOF ADC [AU]");
            h_FTOF.attr().setLineWidth(3);
            h_FTOF.attr().setTitle("Class " + Arrays.toString(names.get(classs)));
            H1F h2_FTOF = new H1F("E", 31, 0, 31);
            h2_FTOF.attr().setLineColor(classs+2);// tags start at 1
            h2_FTOF.attr().setTitleX("N Hits in FTOF");
            h2_FTOF.attr().setLineWidth(3);
            h2_FTOF.attr().setTitle("Class " + Arrays.toString(names.get(classs)));

            H1F h_HTCC = new H1F("E_HTCC", 101, -0.01, 1);
            h_HTCC.attr().setLineColor(classs+2);// tags start at 1
            h_HTCC.attr().setTitleX("HTCC ADC [AU]");
            h_HTCC.attr().setLineWidth(3);
            h_HTCC.attr().setTitle("Class " + Arrays.toString(names.get(classs)));
            H1F h2_HTCC = new H1F("E", 21, 0, 21);
            h2_HTCC.attr().setLineColor(classs+2);// tags start at 1
            h2_HTCC.attr().setTitleX("N Hits in HTCC");
            h2_HTCC.attr().setLineWidth(3);
            h2_HTCC.attr().setTitle("Class " + Arrays.toString(names.get(classs)));

            for (int j = 0; j < file_arr.length; j++) {
                String file = file_arr[j]+"_daq.h5";
                HipoReader r = new HipoReader();
            
                r.open(file);

                System.out.println("Reading file: "+file);

                int nMax = max/file_arr.length;

                CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
                CompositeNode nFTOF = new CompositeNode( 13, 3,  "bbsbifs", 4096);
                CompositeNode nHTCC = new CompositeNode( 14, 5, "bbsbifs", 4096);

                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray FTOFArray = Nd4j.zeros(nMax, 62);
                INDArray HTCCArray = Nd4j.zeros(nMax, 8);
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);
                    event.read(nEC, 11, 2);
                    event.read(nFTOF, 13, 3);
                    event.read(nHTCC, 14, 5);
                    Node node = event.read(5, 4);
                    int[] ids = node.getInt();

                    List<Double> energies = new ArrayList<Double>();
                    Level3Utils.fillEC(ECArray, nEC, ids[2], counter, energies);
                    for (double energy : energies) {
                        h.fill(energy);
                    }
                    h2.fill(energies.size());

                    Level3Utils.fillFTOF(FTOFArray,nFTOF,ids[2],counter);
                    int nFTOF_Hits=0;
                    for(int l=0;l<62;l++){
                        if(FTOFArray.getFloat(counter, l)>0){
                            nFTOF_Hits++;
                            h_FTOF.fill(FTOFArray.getFloat(counter, l));
                        }
                    }
                    h2_FTOF.fill(nFTOF_Hits);
                    Level3Utils.fillHTCC(HTCCArray,nHTCC,ids[2],counter);
                    int nHTCC_Hits=0;
                    for(int l=0;l<8;l++){
                        if(HTCCArray.getFloat(counter, l)>0){
                            nHTCC_Hits++;
                            h_HTCC.fill(HTCCArray.getFloat(counter, l));
                        }
                    }
                    h2_HTCC.fill(nHTCC_Hits);
                    counter++;

                }
            }
            if (classs == 0) {
                c.draw(h);
                c2.draw(h2);
                c_FTOF.draw(h_FTOF);
                c2_FTOF.draw(h2_FTOF);
                c_HTCC.draw(h_HTCC);
                c2_HTCC.draw(h2_HTCC);
            } else{
                c.draw(h,"same");
                c2.draw(h2,"same");
                c_FTOF.draw(h_FTOF,"same");
                c2_FTOF.draw(h2_FTOF,"same");
                c_HTCC.draw(h_HTCC,"same");
                c2_HTCC.draw(h2_HTCC,"same");
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
                CompositeNode nFTOF = new CompositeNode( 13, 3,  "bbsbifs", 4096);
                CompositeNode nHTCC = new CompositeNode( 14, 5, "bbsbifs", 4096);

                INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray FTOFArray = Nd4j.zeros(nMax, 62);
                INDArray HTCCArray = Nd4j.zeros(nMax, 8);
                INDArray OUTArray = Nd4j.zeros(nMax, files.size());
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);

                    event.read(nDC, 12, 1);
                    event.read(nEC, 11, 2);
                    event.read(nFTOF, 13, 3);
                    event.read(nHTCC, 14, 5);

                    Node node = event.read(5, 4);

                    int[] ids = node.getInt();

                    // System.out.printf("event tag (%d) & ID (%d)\n", ids[1], ids[0]);
                    // System.out.printf("event tag (%d) & ID (%d)\n",event.getEventTag(),ids[0]);

                    Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                    int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                    Level3Utils.fillLabels_MultiClass(OUTArray, files.size(), classs, counter);// tag
                    Level3Utils.fillFTOF(FTOFArray,nFTOF,ids[2],counter);
                    Level3Utils.fillHTCC(HTCCArray,nHTCC,ids[2],counter);
                    counter++;

                }
                System.out.printf("loaded samples (%d)\n\n\n", counter);
                File fileEC = new File(out + "EC_" + String.valueOf(names.get(classs)[j]) + ".npy");
                File fileDC = new File(out + "DC_" + String.valueOf(names.get(classs)[j]) + ".npy");
                File fileFTOF = new File(out + "FTOF_" + String.valueOf(names.get(classs)[j]) + ".npy");
                File fileHTCC = new File(out + "HTCC_" + String.valueOf(names.get(classs)[j]) + ".npy");
                try {
                    Nd4j.writeAsNumpy(ECArray, fileEC);
                    Nd4j.writeAsNumpy(DCArray, fileDC);
                    Nd4j.writeAsNumpy(FTOFArray, fileFTOF);
                    Nd4j.writeAsNumpy(HTCCArray, fileHTCC);
                } catch (IOException e) {
                    System.out.println("Could not write file");
                }
                added_classes++;
            }
            classs++;
        }

    }

    public MultiDataSet getClassesFromFile(List<String[]> files,List<String[]> names, int max,double trainTestP) {
        INDArray[] inputs = new INDArray[4];
        INDArray[] outputs = new INDArray[1];
        //added tag is for individual tag
        //classs is for each array of tags ie class
        int classs=0;

        for (String[] file_arr : files) {

            System.out.printf("Class: %d",classs);

            INDArray[] inputs_class = new INDArray[4];
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
                CompositeNode nFTOF = new CompositeNode( 13, 3,  "bbsbifs", 4096);
                CompositeNode nHTCC = new CompositeNode( 14, 5, "bbsbifs", 4096);

                INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
                INDArray FTOFArray = Nd4j.zeros(nMax, 1,62,1);
                INDArray HTCCArray = Nd4j.zeros(nMax, 1,8,1);
                INDArray OUTArray = Nd4j.zeros(nMax, files.size());
                Event event = new Event();
                int counter = 0,eventNb=0;
                while (r.hasNext() == true && counter < nMax) {
                    r.nextEvent(event);

                    event.read(nDC, 12, 1);
                    event.read(nEC, 11, 2);
                    event.read(nFTOF, 13, 3);
                    event.read(nHTCC, 14, 5);

                    Node node = event.read(5, 4);

                    int[] ids = node.getInt();

                    // System.out.printf("event tag (%d) & ID (%d)\n", ids[1], ids[0]);
                    // System.out.printf("event tag (%d) & ID (%d)\n",event.getEventTag(),ids[0]);

                    //allows us to keep N last events for testing
                    if (eventNb >= start) {
                        //Level3Utils.fillDC_wLayers(DCArray, nDC, ids[2], counter);
                        Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                        int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                        Level3Utils.fillLabels_MultiClass(OUTArray, files.size(), classs, counter);// tag
                        Level3Utils.fillFTOF(FTOFArray,nFTOF,ids[2],counter);
                        Level3Utils.fillHTCC(HTCCArray,nHTCC,ids[2],counter);
                        counter++;
                    }
                    eventNb++;

                }

                System.out.printf("loaded samples (%d)\n\n\n", counter);
                if (added_classes == 0) {
                    //inputs = new INDArray[] { DCArray, ECArray };
                    inputs_class = new INDArray[] { DCArray, ECArray,FTOFArray,HTCCArray };
                    outputs_class = new INDArray[] { OUTArray };
                } else {
                    inputs_class[0] = Nd4j.vstack(inputs_class[0], DCArray);
                    inputs_class[1] = Nd4j.vstack(inputs_class[1], ECArray);
                    //remove if not using HTCC, FTOF
                    inputs_class[2] = Nd4j.vstack(inputs_class[2], FTOFArray);
                    inputs_class[3] = Nd4j.vstack(inputs_class[3], HTCCArray);
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
                MultiDataSet datasetEC = new MultiDataSet(new INDArray[] { inputs_class[1] },
                        new INDArray[] { inputs_class[1] });
                datasetEC.shuffle();
                datasetEC.shuffle();
                MultiDataSet datasetFTOF = new MultiDataSet(new INDArray[] { inputs_class[2] },
                        new INDArray[] { inputs_class[2] });
                datasetFTOF.shuffle();
                datasetFTOF.shuffle();
                datasetFTOF.shuffle();
                MultiDataSet datasetHTCC = new MultiDataSet(new INDArray[] { inputs_class[3] },
                        new INDArray[] { inputs_class[3] });
                datasetHTCC.shuffle();
                datasetHTCC.shuffle();
                datasetHTCC.shuffle();
                datasetHTCC.shuffle();
                inputs_class[0] = datasetDC.getFeatures()[0];
                inputs_class[1] = datasetEC.getFeatures()[0];
                inputs_class[2] = datasetFTOF.getFeatures()[0];
                inputs_class[3] = datasetHTCC.getFeatures()[0];
                // Note: OUTArray should be the same at all rows so it doesn't matter
                // if it isn't ordered the same as other arrays
            }

            if (classs == 0) {
                // inputs = new INDArray[] { DCArray, ECArray };
                inputs = inputs_class;
                outputs = outputs_class;
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], inputs_class[0]);
                inputs[1] = Nd4j.vstack(inputs[1], inputs_class[1]);
                // remove if not using HTCC, FTOF
                inputs[2] = Nd4j.vstack(inputs[2],inputs_class[2]);
                inputs[3] = Nd4j.vstack(inputs[3], inputs_class[3]);
                outputs[0] = Nd4j.vstack(outputs[0], outputs_class[0]);
            }
            classs++;  

        }
        
        MultiDataSet dataset = new MultiDataSet(inputs,outputs);
        dataset.shuffle();
        return dataset;
    }

    public static void main(String[] args) {
     
        String dir = "/Users/tyson/data_repo/trigger_data/sims/";
        //String out = "/Users/tyson/data_repo/trigger_data/sims/python/";

        //String dir = "/scratch/clasrun/caos/sims/";

        List<String[]> files = new ArrayList<>();
        /*files.add(new String[] { dir+"pim"});
        files.add(new String[] { dir+"gamma"});
        files.add(new String[] { dir+"pos"});
        files.add(new String[] {dir+"pim",dir+"pos",dir+"el",dir+"gamma"});*/
        files.add(new String[] { dir+"el" });

        /*files.add(new String[] { dir+"pim", dir+"gamma",dir+"pim"});// ,dir+"pos"});//,dir+"pim"});
        files.add(new String[] { dir+"el" });*/

        List<String[]> names = new ArrayList<>();
        /*names.add(new String[] { "pim"});
        names.add(new String[] { "gamma"});
        names.add(new String[] { "pos" });
        names.add(new String[]{"mixMatch","mixMatch","mixMatch","mixMatch"});*/
        names.add(new String[] { "el" });

        /*names.add(new String[] { "pim", "gamma","mixMatch"});// ,"pos"});//,"mixMatch"});
        names.add(new String[] { "el" });*/

        String net = "0d_FTOFHTCC"; //"0d_allLayers"
        Level3Trainer_Simulation t = new Level3Trainer_Simulation();

        //t.getEnergiesForClassesFromFiles(files,names, 10000);
        //t.histClasses(files,names, 10000);
        //t.saveClasses(files,names, out, 50000);
        t.plotHTCC(files,names, 10000);

        t.cnnModel = net;

        // if not transfer learning
        //t.initNetwork(files.size());

        // transfer learning
        // t.load("level3_sim_"+net+".network");

        /*t.nEpochs = 750;//500
        t.trainFile(files,names,30000,5000,10000);//30000 5000 10000
        t.save("level3_sim_MC_wMixMatch");

        t.load("level3_sim_MC_wMixMatch_"+net+".network");
        t.evaluateFile(files,names,5000,false);//5000*/

    }
}
