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


    public void evaluateFile(List<String[]> files,List<String[]> names,String bg, Boolean addBG, int nEvents_pSample, Boolean doPlots) {

        MultiDataSet data = this.getClassesFromFile(files,names,bg,addBG,nEvents_pSample,0.9);

        long nTestEvents = data.getFeatures()[0].shape()[0];

        //plotDCExamples(data.getFeatures()[0], 5,0);

        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1]);
        //0d_FTOFHTCC
        //INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[2], data.getFeatures()[3]);
        //0f
        INDArray[] outputs = network.output(data.getFeatures()[0], data.getFeatures()[1], data.getFeatures()[3]);


        int el_index=findIndexClassString(files,"el");

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics_MultiClass metrics = new Level3Metrics_MultiClass(nTestEvents, outputs[0], data.getLabels()[0],el_index,files.size(),doPlots);

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

    public void trainFile(List<String[]> files,List<String[]> names,String bg, Boolean addBG, int nEvents_pSample, int nEvents_pSample_test,int batchSize) {

        MultiDataSet data = this.getClassesFromFile(files,names,bg,addBG,nEvents_pSample,0);
        MultiDataSet data_test = this.getClassesFromFile(files,names,bg,addBG,nEvents_pSample_test,0.9);

        long NTotEvents = data.getFeatures()[0].shape()[0];
        long NTotEvents_test = data_test.getFeatures()[0].shape()[0];

        Evaluation eval = new Evaluation(files.size());

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
                //0d_FTOFHTCC
                //network.fit(new INDArray[] {DC_b,EC_b,FTOF_b,HTCC_b}, new INDArray[] {Lab_b});
                //0f
                network.fit(new INDArray[] {DC_b,EC_b,HTCC_b}, new INDArray[] {Lab_b});
            }

            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);
            //INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1]);
            //0d_FTOFHTCC
            //INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1], data_test.getFeatures()[2], data_test.getFeatures()[3]);
            //0f
            INDArray[] outputs = network.output(data_test.getFeatures()[0], data_test.getFeatures()[1], data_test.getFeatures()[3]);
            
		    eval.eval(data_test.getLabels()[0], outputs[0]);
            System.out.printf("Test Purity: %f, Efficiency: %f\n",eval.precision(),eval.recall());
            if (i % 50 == 0 && i != 0) {
                this.save("tmp_models/level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
        
    }

    public INDArray add3DArrays(INDArray arr1, INDArray arr2) {
        if (arr1.equalShapes(arr2)) {
            for (int i = 0; i < arr1.shape()[0]; i++) {
                for (int k = 0; k < arr1.shape()[1]; k++) {
                    for (int l = 0; l < arr1.shape()[2]; l++) {
                        // if no entry in array 1 and there's an entry in array 2, then add it to array 1
                        if (arr1.getFloat(i, k, l) == 0 && arr2.getFloat(i, k, l) != 0) {
                            arr1.putScalar(new int[] { i, k, l }, arr2.getFloat(i, k, l));
                        }
                        // if there is an entry in array 1, we keep it
                        // never add array 2 to array 1 if array 1 already has an entry
                    }
                }
            }
        } else {
            System.out.println("****** Array shapes don't match, returning first array ******");
        }
        return arr1;

    }

    public static INDArray addInputArrays(INDArray arr1, INDArray arr2){
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

    public static MultiDataSet getBg(String bgLoc, int max,int start) {
       
        int added = 0;

        // INDArray DCArray = Nd4j.zeros(max, 1, 6, 112);
        INDArray DCArray = Nd4j.zeros(max, 6, 6, 112);
        INDArray ECArray = Nd4j.zeros(max, 1, 6, 72);
        INDArray FTOFArray = Nd4j.zeros(max, 1, 62, 1);
        INDArray HTCCArray = Nd4j.zeros(max, 1, 8, 1);
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
            CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
            CompositeNode nFTOF = new CompositeNode(13, 3, "bbsbifs", 4096);
            CompositeNode nHTCC = new CompositeNode(14, 5, "bbsbifs", 4096);

            Event event = new Event();
            int counter = 0;
            while (r.hasNext() == true && counter < nMax) {
                r.nextEvent(event);

                event.read(nDC, 12, 1);
                event.read(nEC, 11, 2);
                event.read(nFTOF, 13, 3);
                event.read(nHTCC, 14, 5);

                Node node = event.read(5, 4);

                int[] ids = node.getInt();

                // Level3Utils.fillDC_wLayers(DCArray, nDC, ids[2], counter);
                // Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                Level3Utils.fillDC_SepSL(DCArray, nDC, ids[2], counter);
                int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                Level3Utils.fillFTOF(FTOFArray, nFTOF, ids[2], counter);
                Level3Utils.fillHTCC(HTCCArray, nHTCC, ids[2], counter);

                counter++;
                added++;
            }
        }
        return new MultiDataSet(new INDArray[]{DCArray,ECArray,FTOFArray,HTCCArray},new INDArray[]{});
    }

    public static MultiDataSet addBg(String bgLoc, int max,int start,MultiDataSet dataset) {

        MultiDataSet bgDataSet=getBg(bgLoc, max, start);

        //inputs_class = new INDArray[] { DCArray, ECArray,FTOFArray,HTCCArray };
        INDArray[] inputs = new INDArray[4];//1 with only DC
        INDArray[] outputs = new INDArray[1];
        inputs[0]=addInputArrays(dataset.getFeatures()[0],bgDataSet.getFeatures()[0]);
        inputs[1]=addInputArrays(dataset.getFeatures()[1],bgDataSet.getFeatures()[1]);
        inputs[2]=addInputArrays(dataset.getFeatures()[2],bgDataSet.getFeatures()[2]);
        inputs[3]=addInputArrays(dataset.getFeatures()[3],bgDataSet.getFeatures()[3]);
        outputs[0]=dataset.getLabels()[0];
        dataset = new MultiDataSet(inputs,outputs);
        return dataset;
    }

    public MultiDataSet getClassesFromFile(List<String[]> files,List<String[]> names,String bg,Boolean addBG, int max,double trainTestP) {
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
                String file = file_arr[j]+"_daq_v3.h5";
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

                //INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
                INDArray DCArray = Nd4j.zeros(nMax, 6, 6, 112);
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
                        //Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                        Level3Utils.fillDC_SepSL(DCArray, nDC, ids[2], counter);
                        int nHits = Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                        Level3Utils.fillLabels_MultiClass(OUTArray, files.size(), classs, counter);
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

            //corrupt portion of data by removing certain superlayers
            if((names.get(classs)[0] != "gamma") && (names.get(classs)[0] != "empty")){
                long nEv=Math.round(inputs_class[0].shape()[0]);
                int fileMtp=1;
                if(max>10000){fileMtp=(int) Math.ceil(max/10000);}
                MultiDataSet bgDataSet=new MultiDataSet();
                if(bg!=""){
                    bgDataSet=getBg(bg,(int) nEv,(int) Math.round(trainTestP*100)+classs*fileMtp+1);
                }
                
                Random rand = new Random();
                for(int i=0;i<nEv;i++){
                    //SLs to skip
                    int SLs1 = rand.nextInt(6);
                    int SLs2 = SLs1;
                    while(SLs1==SLs2){SLs2=rand.nextInt(6);}
                    
                    //corrupt only a portion of data
                    if (i < (nEv / 2)) {
                        //System.out.printf("Nev/2 %d, i %d \n",(nEv/2),i);
                        if (addBG) {
                            INDArray to_rm1 = inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs1),
                                    NDArrayIndex.all(),
                                    NDArrayIndex.all()).dup();
                            INDArray to_rm2 = inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs2),
                                    NDArrayIndex.all(),
                                    NDArrayIndex.all()).dup();
                            INDArray wbg = add3DArrays(
                                    inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.all()).dup(),
                                    bgDataSet.getFeatures()[0].get(NDArrayIndex.point(i), NDArrayIndex.all(),
                                            NDArrayIndex.all(),
                                            NDArrayIndex.all()).dup());
                            INDArray wbg_1 = wbg.get(NDArrayIndex.point(SLs1), NDArrayIndex.all(),
                                    NDArrayIndex.all()).dup();
                            INDArray wbg_2 = wbg.get(NDArrayIndex.point(SLs2), NDArrayIndex.all(),
                                    NDArrayIndex.all()).dup();
                            INDArray Cleaned1 = wbg_1.sub(to_rm1);
                            INDArray Cleaned2 = wbg_2.sub(to_rm2);
                            wbg.get(NDArrayIndex.point(SLs1), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Cleaned1);
                            wbg.get(NDArrayIndex.point(SLs2), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(Cleaned2);
                            inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(wbg);
                        } else {
                            if(bg!=""){
                                //for clasdis data, bg is already in data so we don't to add bg again
                                //however we don't just want empty superlayer so we load some other bg for
                                //superlayer and add that
                                inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs1), NDArrayIndex.all(),
                                        NDArrayIndex.all()).assign(bgDataSet.getFeatures()[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs1),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all()));
                                inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs2), NDArrayIndex.all(),
                                        NDArrayIndex.all()).assign(bgDataSet.getFeatures()[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs2),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all()));
                            }
                            else{
                                inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs1), NDArrayIndex.all(),
                                        NDArrayIndex.all()).assign(Nd4j.zeros(6, 112));
                                inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.point(SLs2), NDArrayIndex.all(),
                                        NDArrayIndex.all()).assign(Nd4j.zeros(6, 112));
                            }
                            
                            
                        }
                    } else {// have to add noise to rest of DC data
                        if (addBG) {
                            INDArray wbg = add3DArrays(
                                    inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
                                            NDArrayIndex.all()).dup(),
                                    bgDataSet.getFeatures()[0].get(NDArrayIndex.point(i), NDArrayIndex.all(),
                                            NDArrayIndex.all(),
                                            NDArrayIndex.all()).dup());
                            inputs_class[0].get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.all()).assign(wbg);
                        }
                    }
                    
                }

                if (addBG) {
                    inputs_class[1] = addInputArrays(inputs_class[1], bgDataSet.getFeatures()[1]);
                    inputs_class[2] = addInputArrays(inputs_class[2], bgDataSet.getFeatures()[2]);
                    inputs_class[3] = addInputArrays(inputs_class[3], bgDataSet.getFeatures()[3]);
                }

            } else{
                if (addBG) {
                    int fileMtp=1;
                    if(max>10000){fileMtp=(int) Math.ceil(max/10000);}
                    MultiDataSet bgDataSet=getBg(bg, max, (int) Math.round(trainTestP*100)+classs*fileMtp+1);
                    inputs_class[0] = addInputArrays(inputs_class[0], bgDataSet.getFeatures()[0]);
                    inputs_class[1] = addInputArrays(inputs_class[1], bgDataSet.getFeatures()[1]);
                    inputs_class[2] = addInputArrays(inputs_class[2], bgDataSet.getFeatures()[2]);
                    inputs_class[3] = addInputArrays(inputs_class[3], bgDataSet.getFeatures()[3]);
                }
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
     
        //String dir = "/Users/tyson/data_repo/trigger_data/sims/";
        //String out = "/Users/tyson/data_repo/trigger_data/sims/python/";

        String dir = "/scratch/clasrun/caos/sims/";

        // for clasdis data, bg is already in data so we don't to add bg again
        // however when corrupting we don't want just empty array so 
        // we add data from bg merging
        // think we should do this for real data too
        // code expects addBG false but a non empty bg string
        // obviously if we want to addBG code expects non empty bg string
        String bg=dir+"bg_50nA_10p6/";//"";
        Boolean addBG=false;

        List<String[]> files = new ArrayList<>();
        files.add(new String[] { dir+"claspyth_pim"});
        files.add(new String[] { dir+"claspyth_gamma"});
        //files.add(new String[] { dir+"pos"}); //not even e+ in claspyth data
        files.add(new String[] { dir+"claspyth_pip"});
        //files.add(new String[] {dir+"claspyth_pim",dir+"claspyth_pip"});//,dir+"pos",dir+"el"
        files.add(new String[] { dir+"claspyth_empty" });
        files.add(new String[] { dir+"claspyth_el" });

        /*files.add(new String[] { dir+"claspyth_empty" });
        files.add(new String[]{dir+"claspyth_pim",dir+"claspyth_pip",dir+"claspyth_gamma"});
        files.add(new String[] { dir+"claspyth_el" });*/

        /*files.add(new String[] { dir+"pim", dir+"gamma",dir+"pim"});// ,dir+"pos"});//,dir+"pim"});
        files.add(new String[] { dir+"el" });*/

        List<String[]> names = new ArrayList<>();
        names.add(new String[] { "pim"});
        names.add(new String[] { "gamma"});
        //names.add(new String[] { "pos" });
        names.add(new String[] { "pip" });
        //names.add(new String[]{"mixMatch","mixMatch"});//,"mixMatch","mixMatch"
        names.add(new String[]{"empty",""});//,"mixMatch","mixMatch"
        names.add(new String[] { "el" });
        
        /*names.add(new String[] { "empty" });
        names.add(new String[] { "pim","pip","gamma" });
        names.add(new String[] { "el" });*/

        /*names.add(new String[] { "pim", "gamma","mixMatch"});// ,"pos"});//,"mixMatch"});
        names.add(new String[] { "el" });*/

        String net = "0f"; //"0d_allLayers"
        Level3Trainer_Simulation t = new Level3Trainer_Simulation();

        //t.getEnergiesForClassesFromFiles(files,names, 10000);
        //t.histClasses(files,names, 10000);
        //t.saveClasses(files,names, out, 50000);
        //t.plotHTCC(files,names, 10000);

        t.cnnModel = net;

        // if not transfer learning
        t.initNetwork(files.size());

        // transfer learning
        // t.load("level3_sim_"+net+".network");

        t.nEpochs = 400;//500 //750
        
        t.trainFile(files,names,bg,addBG,30000,1000,1000);//30000 5000 10000
        t.save("level3_sim_MC_wCorrupt_wbg_wEmpty_SIDIS");

        t.load("level3_sim_MC_wCorrupt_wbg_wEmpty_SIDIS_"+net+".network");
        t.evaluateFile(files,names,bg,addBG,1000,false);//5000

    }
}
