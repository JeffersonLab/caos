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
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import twig.data.GraphErrors;
import twig.server.HttpDataServer;
import twig.server.HttpServerConfig;

/**
 *
 * @author gavalian
 */
public class Level3Trainer_MultiClass {

    ComputationGraph network = null;
    public int nEpochs = 25;
    public String cnnModel = "0a";

    public Level3Trainer_MultiClass() {

    }

    public void initNetwork() {
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
            Logger.getLogger(Level3Trainer_MultiClass.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public void load(String file) {
        try {
            network = ComputationGraph.load(new File(file), true);
            System.out.println(network.summary());
        } catch (IOException ex) {
            Logger.getLogger(Level3Trainer_MultiClass.class.getName()).log(Level.SEVERE, null, ex);
        }
    }


    public void evaluateFile(String file, int nEvents) {

        INDArray[] inputs = this.getFromFile(file, nEvents);

        INDArray[] outputs = network.output(inputs[0], inputs[1]);

        long nTestEvents = inputs[0].shape()[0];

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics metrics = new Level3Metrics(nTestEvents, outputs[0], inputs[2]);

    }

    public void trainManyFiles(List<String> files, int nEvents) {

        // INDArray[] inputs = this.getFromFile(files.get(0), nEvents);

        INDArray[] inputs = new INDArray[3];
        int count = 0;
        for (String file : files) {
            INDArray[] inputs_temp = this.getFromFile(file, nEvents);
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


    public INDArray[] getFromFile(String file, int max) {
        HipoReader r = new HipoReader(file);

        int nMax = max;

        if (r.entries() < max)
            nMax = r.entries();

        CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
        CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

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
                //if (ids[2] > 0 && npos<nneg) {
                Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                Level3Utils.fillEC(ECArray, nEC, ids[2], counter);
                Level3Utils.fillLabels(OUTArray, 1, counter);
                counter++;
                npos++;
                //}
            } else {
                // always have more neg than pos, balance dataset by only adding neg after pos
                // do we care if trigger is wrong (&& ids[1] == ids[2])? - means training in
                // worse case scenario
                if (nneg < npos) {
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

    public static void main(String[] args) {
        int mode = -1;

        if (mode > 0) {

            Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();
            t.load("level3_model_0a_1000_epochs.network_0a.network");
            // t.load("level3_model_0b_625_epochs.network_0b.network");
            String file = "rec_clas_005197.evio.00405-00409.hipo_daq.h5";
            t.evaluateFile(file, 10000);

        } else if(mode<0){
            //String baseLoc="/Users/tyson/data_repo/trigger_data/rga/daq_";
            String baseLoc="/Users/tyson/data_repo/trigger_data/rgd/018437/daq_MC_";
            String net="0b";
	        Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();

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
	        //t.trainManyFiles(files,200000);//10
	        //t.save("level3_MC");
	    
	        String file2=baseLoc+"5.h5";

	        t.load("level3_MC_"+net+"_fCF.network");
            //t.load("etc/networks/network-level3-0c-rgc.network");
	        t.evaluateFile(file2,200000);

        }else {

            OptionParser parser = new OptionParser("trainer");

            parser.addOption("-n", "level3", "output network name");
            parser.addOption("-m", "0b", "model name (0a,0b or 0c)");
            parser.addOption("-e", "1024", "number of epochs");
            parser.addOption("-max", "200000", "number of data samples");

            parser.parse(args);

            String net = parser.getOption("-m").stringValue();
            Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();

            t.cnnModel = net;
            t.initNetwork();
            t.nEpochs = parser.getOption("-e").intValue();
            int max = parser.getOption("-max").intValue();

            t.trainManyFiles(parser.getInputList(), max);
            t.save(parser.getOption("-n").stringValue());

        }
    }
}
