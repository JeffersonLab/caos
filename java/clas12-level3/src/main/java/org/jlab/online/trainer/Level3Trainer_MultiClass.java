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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import twig.data.GraphErrors;
import twig.data.H1F;
import twig.graphics.TGCanvas;
import twig.server.HttpDataServer;
import twig.server.HttpServerConfig;

/**
 *
 * @author gavalian, tyson
 */
public class Level3Trainer_MultiClass {

    ComputationGraph network = null;
    public int nEpochs = 25;
    public String cnnModel = "0a";

    public Level3Trainer_MultiClass() {

    }

    public void initNetwork(int nClasses) {
        ComputationGraphConfiguration config = Level3Models_MultiClass.getModel(cnnModel,nClasses);
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


    public void evaluateFile(String file, int nEvents_pSample,List<Integer> tags) {

        INDArray[] inputs = this.getTagsFromFile(file,nEvents_pSample,tags);

        INDArray[] outputs = network.output(inputs[0], inputs[1]);

        long nTestEvents = inputs[0].shape()[0];

        // System.out.println("Number of Test Events "+nTestEvents);
        Level3Metrics_MultiClass metrics = new Level3Metrics_MultiClass(nTestEvents, outputs[0], inputs[2],tags,true);

    }

    public void trainFile(String file,String fileTest, int nEvents_pSample,List<Integer> tags) {

        INDArray[] inputs = this.getTagsFromFile(file,nEvents_pSample,tags);
        INDArray[] inputs_test = this.getTagsFromFile(fileTest,nEvents_pSample,tags);

        /*HttpServerConfig config = new HttpServerConfig();
        config.serverPort = 8525;
        HttpDataServer.create(config);
        GraphErrors graph = new GraphErrors("graph");

        HttpDataServer.getInstance().getDirectory().add("/server/training", graph);
        HttpDataServer.getInstance().start();

        // HttpDataServer.getInstance().getDirectory().list();
        HttpDataServer.getInstance().getDirectory().show();*/

        for (int i = 0; i < nEpochs; i++) {
            long then = System.currentTimeMillis();
            network.fit(new INDArray[] { inputs[0], inputs[1] }, new INDArray[] { inputs[2] });
            long now = System.currentTimeMillis();
            System.out.printf(">>> network iteration %8d, score = %e, time = %12d ms\n",
                    i, network.score(), now - then);

            INDArray[] outputs = network.output(inputs_test[0], inputs_test[1]);
            Evaluation eval = new Evaluation(tags.size());
		    eval.eval(inputs[2], outputs[0]);
            System.out.printf("Test Purity: %f, Efficiency: %f\n",eval.precision(),eval.recall());

            //graph.addPoint(i, network.score());
            if (i % 500 == 0 && i != 0) {
                this.save("tmp_models/level3_model_" + this.cnnModel + "_" + i + "_epochs.network");
            }
        }
    }

    public void getEnergiesForTagsFromFile(String file, int max,List<Integer> tags) {
        int added_tags=0;
        //for testing
        TGCanvas c = new TGCanvas();
        c.setTitle("Calorimeter Energy");
        for (int tag : tags) {
            //for testing
            H1F h = new H1F("E", 101, -0.01, 1);
            h.attr().setLineColor(tag+1);// tags start at 1
            h.attr().setTitleX("Calorimeter Energy [AU]");
            h.attr().setLineWidth(3);
            h.attr().setTitle("Tag "+String.valueOf(tag));
            HipoReader r = new HipoReader();
            r.setTags(tag);
            r.open(file);
            int nMax = max;
            if (r.entries() < max)
                nMax = r.entries();
            CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);
            INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
            Event event = new Event();
            int counter = 0;
            while (r.hasNext() == true && counter < nMax) {
                r.nextEvent(event);
                event.read(nEC, 11, 2);
                Node node = event.read(5, 4);
                int[] ids = node.getInt();
                List<Double> energies = new ArrayList<Double>();
                Level3Utils.fillEC(ECArray, nEC, ids[2], counter,energies);
                for(double energy:energies){h.fill(energy);}
                counter++;
            }
            if (added_tags == 0) {
                c.draw(h);
            } else{
                c.draw(h,"same");
            }
            added_tags++;
        }
    }

    public INDArray[] getTagsFromFile(String file, int max,List<Integer> tags) {
        INDArray[] inputs = new INDArray[3];
        int added_tags=0;

        for (int tag : tags) {

            HipoReader r = new HipoReader();
           
            r.setTags(tag);
            r.open(file);

            int nMax = max;

            if (r.entries() < max)
                nMax = r.entries();

            CompositeNode nDC = new CompositeNode(12, 1, "bbsbil", 4096);
            CompositeNode nEC = new CompositeNode(11, 2, "bbsbifs", 4096);

            INDArray DCArray = Nd4j.zeros(nMax, 1, 6, 112);
            INDArray ECArray = Nd4j.zeros(nMax, 1, 6, 72);
            INDArray OUTArray = Nd4j.zeros(nMax, tags.size());
            Event event = new Event();
            int counter = 0;
            while (r.hasNext() == true && counter < nMax) {
                r.nextEvent(event);
                    
                event.read(nDC, 12, 1);
                event.read(nEC, 11, 2);

                Node node = event.read(5, 4);

                int[] ids = node.getInt();

                //System.out.printf("event tag (%d) & ID (%d)\n", ids[1], ids[0]);
                //System.out.printf("event tag (%d) & ID (%d)\n",event.getEventTag(),ids[0]);

                Level3Utils.fillDC(DCArray, nDC, ids[2], counter);
                Level3Utils.fillEC(ECArray, nEC, ids[2], counter);

                Level3Utils.fillLabels_MultiClass(OUTArray, tags, tag, counter);// tag
                counter++;
                

            }

            /*System.out.printf("tag (%d)",tag);
            System.out.print(OUTArray);
            System.out.print("\n\n");*/

            System.out.printf("loaded samples (%d) for tag %d\n\n\n", counter, tag);
            if (added_tags == 0) {
                inputs=new INDArray[] { DCArray, ECArray, OUTArray };
            } else{
                inputs[0] = Nd4j.vstack(inputs[0], DCArray);
                inputs[1] = Nd4j.vstack(inputs[1], ECArray);
                inputs[2] = Nd4j.vstack(inputs[2], OUTArray);
            }
            added_tags++;
        }

        //can't figure this out
        //List<int[]> dimensions = Arrays.asList(new int[]{0}, new int[]{0}, new int[]{0});
        //Nd4j.shuffle(Arrays.asList(inputs[0], inputs[1],inputs[2]), new Random(12345), dimensions);
        //Random r=new Random(12345);
        /*Nd4j.shuffle(inputs[0], new Random(12345), 0);
        Nd4j.shuffle(inputs[1], new Random(12345), 0);
        Nd4j.shuffle(inputs[2], new Random(12345), 0);*/
        
        return inputs;
    }

    public static void main(String[] args) {
        int mode = -1;

        if (mode > 0) {

            List<Integer> tags= new ArrayList<>();
            for(int i=1;i<6;i++){tags.add(i);}

            Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();
            t.load("level3_model_0a_1000_epochs.network_0a.network");
            // t.load("level3_model_0b_625_epochs.network_0b.network");
            String file = "rec_clas_005197.evio.00405-00409.hipo_daq.h5";
            t.evaluateFile(file, 10000,tags);

        } else if(mode<0){
            //String baseLoc="/Users/tyson/data_repo/trigger_data/rga/daq_";
            String file="/Users/tyson/data_repo/trigger_data/rgd/018437_AI/daq_MC_0.h5";
            String file2="/Users/tyson/data_repo/trigger_data/rgd/018437_AI/daq_MC_5.h5";

            List<Integer> tags= new ArrayList<>();
            //for(int i=1;i<8;i++){tags.add(i);}
            //tags.add(7);
            tags.add(4);
            //tags.add(2);
            tags.add(1);
            
            
            String net="0b";
	        Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();

	        t.cnnModel = net;

            //if not transfer learning
	        //t.initNetwork(tags.size());

            //transfer learning
            //t.load("level3_"+net+".network");

	        //t.nEpochs = 10;
	        //t.trainFile(file,file2,10000,tags);//10
	        //t.save("level3_MC");

            //t.load("level3_MC_"+net+".network");
            //t.load("level3_MC_2C_t4t1_"+net+"_AI.network");
            t.load("level3_MC_2C_test_"+net+"_AI.network");
	        t.evaluateFile(file2,10000,tags);

        }else {

            OptionParser parser = new OptionParser("trainer");

            parser.addOption("-n", "level3", "output network name");
            parser.addOption("-m", "0b", "model name (0a,0b or 0c)");
            parser.addOption("-e", "1024", "number of epochs");
            parser.addOption("-max", "200000", "number of data samples");
            parser.addOption("-f","/Users/tyson/data_repo/trigger_data/rgd/018437/daq_MC_0.h5","input file");

            parser.parse(args);

            String net = parser.getOption("-m").stringValue();
            Level3Trainer_MultiClass t = new Level3Trainer_MultiClass();

            t.cnnModel = net;
            t.initNetwork(5); //nb change this to nb tags once figure out how to pass this in option parser
            t.nEpochs = parser.getOption("-e").intValue();
            int max = parser.getOption("-max").intValue();

            //NB can't figure out how to use option parser for this
            //t.trainFile(parser.getOption("-f").stringValue(), max,parser.getInputList());
            t.save(parser.getOption("-n").stringValue());

        }
    }
}
