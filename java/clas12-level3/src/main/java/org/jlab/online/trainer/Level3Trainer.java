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
import j4np.utils.io.TextFileWriter;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import twig.data.GraphErrors;
import twig.server.HttpDataServer;
import twig.server.HttpServerConfig;

/**
 *
 * @author gavalian
 */
public class Level3Trainer {

    ComputationGraph network = null;
    public int nEpochs = 25;
    public String cnnModel = "0a";

    public Level3Trainer() {

    }

    public void initNetwork() {
        /*
         * ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
         * //.l2(0.0005)
         * .weightInit(WeightInit.XAVIER)
         * .updater(new Adam(1e-3))
         * .graphBuilder()
         * .addInputs("dc", "ec")
         * .addLayer("L1", new ConvolutionLayer.Builder(2,2)
         * .nIn(1)
         * .nOut(6)
         * .activation(Activation.RELU)
         * .stride(1,1).build()
         * , "dc")
         * .addLayer("L2", new ConvolutionLayer.Builder(2,2)
         * .nIn(1)
         * .nOut(6)
         * .activation(Activation.RELU)
         * .stride(1,1).build()
         * , "ec")
         * .addLayer("dcDense", new
         * DenseLayer.Builder().nIn(3330).nOut(48).dropOut(0.5).build(), "L1")
         * .addLayer("ecDense", new
         * DenseLayer.Builder().nIn(2130).nOut(48).dropOut(0.5).build(), "L2")
         * .addVertex("merge", new MergeVertex(), "dcDense", "ecDense")
         * .addLayer("out", new OutputLayer.Builder()
         * .nIn(48+48).nOut(2)
         * .activation(Activation.SOFTMAX)
         * .build()
         * , "merge")
         * .setOutputs("out")
         * .setInputTypes(InputType.convolutional(6, 112, 1),InputType.convolutional(6,
         * 72, 1))
         * .build();
         */
        ComputationGraphConfiguration config = Level3Models.getModel(cnnModel);
        network = new ComputationGraph(config);
        network.init();
        System.out.println(network.summary());
    }

    public void save(String file) {

        try {
            network.save(new File(file + "_" + cnnModel + ".network"));
            System.out.println("saved file : " + file + "\n" + "done...");
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void evaluateFile_out(String file, int nEvents) {
        INDArray[] inputs = this.getFromFile(file, nEvents);
        INDArray[] outputs = network.output(inputs[0], inputs[1]);

        TextFileWriter w = new TextFileWriter();
        w.open("evaluationOutput.csv");
        for (int i = 0; i < nEvents * 6; i++) {
            String output = String.format("%f,%f,%f,%f",
                    inputs[2].getDouble(new int[] { i, 0 }),
                    inputs[2].getDouble(new int[] { i, 1 }),
                    outputs[0].getDouble(new int[] { i, 0 }),
                    outputs[0].getDouble(new int[] { i, 1 }));
            w.writeString(output);
        }
        w.close();
    }

    public void evaluateFile(String file, int nEvents) {

        INDArray[] inputs = this.getFromFile(file, nEvents);

        // inputs=balanceDataset(inputs,nEvents);

        INDArray[] outputs = network.output(inputs[0], inputs[1]);

        long nTestEvents = inputs[0].shape()[0];

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics metrics = new Level3Metrics(nTestEvents, outputs[0], inputs[2]);

    }

    public void evaluateFileNuevo(String file, int nEvents) {

        INDArray[] inputs = this.getFromFileNuevo(file, nEvents);

        // inputs=balanceDataset(inputs,nEvents);

        INDArray[] outputs = network.output(inputs[0], inputs[1]);

        long nTestEvents = inputs[0].shape()[0];

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics metrics = new Level3Metrics(nTestEvents, outputs[0], inputs[2]);

    }

    public void train() {
        /*
         * BackpropagationTrainer trainer = network.getTrainer();
         * trainer.setLearningRate(0.001f) // za ada delta 0.00001f za rms prop 0.001
         * //trainer.setLearningRate(0.01f) // za ada delta 0.00001f za rms prop 0.001
         * .setMaxError(0.0001f)
         * .setOptimizer(OptimizerType.SGD) // use adagrad optimization algorithm
         * .setL1Regularization(0.005f)
         * //.setL2Regularization(0.001f)
         * //.setBatchSize(200)
         * //.setBatchMode(true)
         * .setMaxEpochs(this.nEpochs);
         */
        INDArray[] inputs = getDummyInputs(1000);
        for (int i = 0; i < 1000; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
        }
    }

    public void trainFile(String file, int nEvents) {

        INDArray[] inputs = this.getFromFile(file, nEvents);

        inputs = balanceDataset(inputs, nEvents);

        HttpServerConfig config = new HttpServerConfig();
        config.serverPort = 8525;
        HttpDataServer.create(config);
        GraphErrors graph = new GraphErrors("graph");

        HttpDataServer.getInstance().getDirectory().add("/server/training", graph);
        HttpDataServer.getInstance().start();

        // HttpDataServer.getInstance().getDirectory().list();
        HttpDataServer.getInstance().getDirectory().show();

        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
            graph.addPoint(i, network.score());
            if (i % 25 == 0 && i != 0) {
                this.save("level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
        // network.output()
    }

    public void transferTrainManyFiles(List<String> files, int nEvents, String networkFile, int max) {
        this.load(networkFile);
        INDArray[] inputs = this.getFromFile(files.get(0), nEvents);
        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
            // graph.addPoint(i, network.score());
            if (i % 25 == 0 && i != 0) {
                this.save("level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
    }

    public void trainManyFiles(List<String> files, int nEvents) {

        INDArray[] inputs = this.getFromFile(files.get(0), nEvents);

        HttpServerConfig config = new HttpServerConfig();
        config.serverPort = 8525;
        HttpDataServer.create(config);
        GraphErrors graph = new GraphErrors("graph");

        HttpDataServer.getInstance().getDirectory().add("/server/training", graph);
        HttpDataServer.getInstance().start();

        // HttpDataServer.getInstance().getDirectory().list();
        HttpDataServer.getInstance().getDirectory().show();

        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
            graph.addPoint(i, network.score());
            if (i % 25 == 0 && i != 0) {
                this.save("level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
    }

    public void trainManyFilesNuevo(List<String> files, int nEvents) {

        // INDArray[] inputs = this.getFromFileNuevo(files.get(0), nEvents);

        INDArray[] inputs = new INDArray[3];
        int count = 0;
        for (String file : files) {
            INDArray[] inputs_temp = this.getFromFileNuevo(file, nEvents);
            if (count == 0) {
                inputs[0] = inputs_temp[0];
                inputs[1] = inputs_temp[1];
                inputs[2] = inputs_temp[2];
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], inputs_temp[0]);
                inputs[1] = Nd4j.vstack(inputs[1], inputs_temp[1]);
                inputs[2] = Nd4j.vstack(inputs[2], inputs_temp[2]);
            }
        }

        HttpServerConfig config = new HttpServerConfig();
        config.serverPort = 8525;
        HttpDataServer.create(config);
        GraphErrors graph = new GraphErrors("graph");

        HttpDataServer.getInstance().getDirectory().add("/server/training", graph);
        HttpDataServer.getInstance().start();

        // HttpDataServer.getInstance().getDirectory().list();
        HttpDataServer.getInstance().getDirectory().show();

        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
            graph.addPoint(i, network.score());
            /*if (i % 25 == 0 && i != 0) {
                this.save("level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }*/
        }
    }

    public void trainManyFiles(String loc, int rangeLow, int rangeHigh, int nEvents) {

        INDArray[] inputs = new INDArray[3];

        for (int i = rangeLow; i < rangeHigh; i++) {

            String file = loc + String.valueOf(i) + ".h5";

            INDArray[] inputs_temp = this.getFromFile(file, nEvents);

            inputs_temp = balanceDataset(inputs_temp, nEvents);

            if (i == 0) {
                inputs[0] = inputs_temp[0];
                inputs[1] = inputs_temp[1];
                inputs[2] = inputs_temp[2];
            } else {
                inputs[0] = Nd4j.vstack(inputs[0], inputs_temp[0]);
                inputs[1] = Nd4j.vstack(inputs[1], inputs_temp[1]);
                inputs[2] = Nd4j.vstack(inputs[2], inputs_temp[2]);
            }
        }

        HttpServerConfig config = new HttpServerConfig();
        config.serverPort = 8525;
        HttpDataServer.create(config);
        GraphErrors graph = new GraphErrors("graph");

        HttpDataServer.getInstance().getDirectory().add("/server/training", graph);
        HttpDataServer.getInstance().start();

        // HttpDataServer.getInstance().getDirectory().list();
        HttpDataServer.getInstance().getDirectory().show();

        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d\n",
                    i, network.score(), now - then);
            graph.addPoint(i, network.score());
            if (i % 25 == 0 && i != 0) {
                this.save("level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }

        // network.output()
    }

    public INDArray[] balanceDataset(INDArray[] inputs, int nEvents) {
        INDArray DCArray_new = Nd4j.zeros(1, 1, 6, 112);
        INDArray ECArray_new = Nd4j.zeros(1, 1, 6, 72);
        INDArray OUTArray_new = Nd4j.zeros(1, 2);

        INDArray DCArray = inputs[0];
        INDArray ECArray = inputs[1];
        INDArray OUTArray = inputs[2];

        int nNegatives = 0;
        int nPositives = 0;
        int nPredictions = 0;

        for (int i = 0; i < nEvents * 6; i++) {

            INDArray DCArray_ev = Nd4j.zeros(1, 1, 6, 112);
            INDArray ECArray_ev = Nd4j.zeros(1, 1, 6, 72);
            INDArray OUTArray_ev = Nd4j.zeros(1, 2);

            DCArray_ev.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(
                    DCArray.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
            ECArray_ev.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()).assign(
                    ECArray.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
            OUTArray_ev.get(NDArrayIndex.point(0), NDArrayIndex.all())
                    .assign(OUTArray.get(NDArrayIndex.point(i), NDArrayIndex.all()));

            // check this row isn't empty
            if (OUTArray.getInt(i, 0) != 0 || OUTArray.getInt(i, 1) != 0) {

                if (nPredictions == 0) {
                    DCArray_new = DCArray_ev;
                    ECArray_new = ECArray_ev;
                    OUTArray_new = OUTArray_ev;
                    if (OUTArray.getInt(i, 0) == 0) {
                        nPositives++;
                    } else {
                        nNegatives++;
                    }
                    nPredictions++;
                } else {
                    if (OUTArray.getInt(i, 0) == 0) {
                        DCArray_new = Nd4j.vstack(DCArray_new, DCArray_ev);
                        ECArray_new = Nd4j.vstack(ECArray_new, ECArray_ev);
                        OUTArray_new = Nd4j.vstack(OUTArray_new, OUTArray_ev);
                        nPredictions++;
                        nPositives++;
                    } else {
                        // probably more negatives than positives
                        // want to have equal proportion
                        if (nNegatives < nPositives) {
                            DCArray_new = Nd4j.vstack(DCArray_new, DCArray_ev);
                            ECArray_new = Nd4j.vstack(ECArray_new, ECArray_ev);
                            OUTArray_new = Nd4j.vstack(OUTArray_new, OUTArray_ev);
                            nPredictions++;
                            nNegatives++;
                        }

                    }
                } // check if first event
            } // check this is row is not empty

        } // loop over event

        System.out.println("Number of Events " + nPredictions + " positive " + nPositives + " negatives " + nNegatives);

        return new INDArray[] { DCArray_new, ECArray_new, OUTArray_new };

    }

    public INDArray[] getFromFileNuevo(String file, int max) {
        HipoReader r = new HipoReader(file);

        int nMax = max;

        if (r.entries() < max)
            nMax = r.entries();

        CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
        CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
        CompositeNode nRC = new CompositeNode(5, 1, "b", 10);
        CompositeNode nET = new CompositeNode(5, 2, "b", 10);

        INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
        INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
        INDArray OUTArray = Nd4j.zeros(nMax, 2);
        Event event = new Event();
        int counter = 0;
        int npos = 0;
        int nneg = 0;
        while (r.hasNext() == true && counter < nMax) {
            r.nextEvent(event);
            event.read(nDC, 12, 1);
            event.read(nEC, 11, 2);

            Node node = event.read(5, 4);

            int[] ids = node.getInt();

            // Do we care if the trigger is fired in the event? (ids[1]>0) - no
            // do we care if trigger is right (&& ids[1] == ids[2])? - no
            if (ids[0] == 11) {
                // Want particle sector to be non null (ids[2]) for positive only
                if (ids[2] > 0) { //V2 data requires  && npos<nneg
                    Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                    Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                    Level3Utils.fillLabels(OUTArray, 1, counter);
                    counter++;
                    npos++;
                }
            } else {
                // always have more neg than pos, balance dataset by only adding neg after pos
                // do we care if trigger is wrong (&& ids[1] == ids[2])? - means training in
                // worse case scenario
                if (nneg < npos) { //MultiClass data requires this
                    Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                    Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                    Level3Utils.fillLabels(OUTArray, 0, counter);
                    counter++;
                    nneg++;
                }
            }

        }

        System.out.printf("\n\n loaded samples (%d)  positive = %s, negative = %d\n", counter, npos, nneg);
        return new INDArray[] { DCArray, ECArray, OUTArray };
    }

    public INDArray[] getFromFile(String file, int max) {
        HipoReader r = new HipoReader(file);

        int nMax = max;

        if (r.entries() < max)
            nMax = r.entries();

        CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
        CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
        CompositeNode nRC = new CompositeNode(5, 1, "b", 10);
        CompositeNode nET = new CompositeNode(5, 2, "b", 10);

        INDArray DCArray = Nd4j.zeros(nMax * 6, 1, 6, 112);
        INDArray ECArray = Nd4j.zeros(nMax * 6, 1, 6, 72);
        INDArray OUTArray = Nd4j.zeros(nMax * 6, 2);
        Event event = new Event();

        // for(int i = 0; i < max; i++){
        int i = 0;
        while (r.hasNext() == true && i < nMax) {
            r.nextEvent(event);
            event.read(nDC, 12, 1);
            event.read(nEC, 11, 2);
            event.read(nRC, 5, 1);
            event.read(nET, 5, 2);
            // System.out.printf(" READ %d %d
            // %d\n",nDC.getRows(),nEC.getRows(),nRC.getRows());
            Level3Utils.fillDC(DCArray, nDC, i);
            Level3Utils.fillEC(ECArray, nEC, i);
            // Level3Utils.fillLabels(OUTArray, nRC, i);
            Level3Utils.fillLabels(OUTArray, nET, i);
            i++;
            System.out.println(" getting " + i + "  out of " + nMax + "  " + r.hasNext());
        }

        return new INDArray[] { DCArray, ECArray, OUTArray };
    }

    public INDArray[] getDummyInputs(int batch) {
        INDArray DCArray = Nd4j.zeros(batch * 6, 1, 6, 112);
        INDArray ECArray = Nd4j.zeros(batch * 6, 1, 6, 72);
        INDArray OUTArray = Nd4j.zeros(batch * 6, 2);
        return new INDArray[] { DCArray, ECArray, OUTArray };
    }

    public static void main(String[] args) {
        int mode = -1;

        if (mode > 0) {

            Level3Trainer t = new Level3Trainer();
            t.load("level3_model_0a_1000_epochs.network_0a.network");
            // t.load("level3_model_0b_625_epochs.network_0b.network");
            String file = "rec_clas_005197.evio.00405-00409.hipo_daq.h5";
            t.evaluateFileNuevo(file, 10000);

        } else if(mode<0){
            String baseLoc="/Users/tyson/data_repo/trigger_data/rga/daq_MC_";
            //String baseLoc="/Users/tyson/data_repo/trigger_data/rgd/018437/daq_MC_";
            String net="0b";
	        Level3Trainer t = new Level3Trainer();

            List<String> files= new ArrayList<>();
            for (int file=0;file<5;file+=5){
                files.add(baseLoc+String.valueOf(file)+".h5");
            }

	        t.cnnModel = net;

            //if not transfer learning
	        //t.initNetwork();

            //transfer learning
            //t.load("level3_"+net+".network");

	        t.nEpochs = 1500;
	        //t.trainManyFilesNuevo(files,200000);//10
	        //t.save("level3");
	    
	        String file2=baseLoc+"5.h5";

	        t.load("level3_"+net+"_fCF_rga.network");
            //t.load("etc/networks/network-level3-0c-rgc.network");
	        t.evaluateFileNuevo(file2,100000);

        }else {

            OptionParser parser = new OptionParser("trainer");

            parser.addOption("-n", "level3", "output network name");
            parser.addOption("-m", "0b", "model name (0a,0b or 0c)");
            parser.addOption("-e", "1024", "number of epochs");
            parser.addOption("-max", "200000", "number of data samples");

            parser.parse(args);

            String net = parser.getOption("-m").stringValue();
            Level3Trainer t = new Level3Trainer();

            t.cnnModel = net;
            t.initNetwork();
            t.nEpochs = parser.getOption("-e").intValue();
            int max = parser.getOption("-max").intValue();

            t.trainManyFilesNuevo(parser.getInputList(), max);
            t.save(parser.getOption("-n").stringValue());

        }
    }
}
