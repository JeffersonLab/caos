/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.level3;
import j4np.data.base.DataFrame;
import j4np.hipo5.data.Event;
import j4np.data.base.DataSource;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;
import j4np.utils.io.OptionParser;
import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public class Level3ProcessFile {
    
    public String networkFile = "etc/networks/level3-default.network";
    private  Level3Processor processor = new Level3Processor();
    private  HipoWriter      writer = null;
    public  int              batchSize = 1024;
    public  int              maxEvents = 100;
    
    private DataFrame<Event>    dataFrame = new DataFrame();
    private DataFrame<Event>  sourceFrame = new DataFrame();
    
    private long timeProcessing = 0L;
    private long countProcessing = 0L;
    
    public Level3ProcessFile(){
        
    }
    
    public void initialize(){
        processor.load(networkFile);
        writer = new HipoWriter();
        writer.setMaxSize(4*1024*1024*1024);
        writer.setCompressionType(0);
        writer.open("output_level3.h5");
        
        for(int i = 0; i < batchSize; i++) sourceFrame.addEvent(new Event());
    }
    
    private void outputBank(CompositeNode node, INDArray[] output, int order){
        float max = 0.0f;
        for(int i = 0; i < 6; i++){            
            float value = output[0].getFloat(order*6+i,1);
            if(value>max) max = value;
            //node.putFloat(0, i+1, value);
            node.putInt(0, 0, 2);
        }
        node.putFloat(0, 0, max);
    }
    
    private void outputBank(float[] node, INDArray[] output, int order){
        float max = 0.0f;
        for(int i = 0; i < 6; i++){            
            float value = output[0].getFloat(order*6+i,1);
            if(value>max) max = value;
            node[ i+1] = value;
        }
        node[0] = max;
    }
    
    public void writeOutput(INDArray[] output, List<Event> events){
        CompositeNode node = new CompositeNode(5,5,"i",32);
        node.setRows(7);
        float[] array = new float[7];
        for(int row = 0; row < events.size(); row++){
            //outputBank(node,output,row);
            outputBank(array,output,row);

            //node.print();
            //System.out.println("---- before writing " + node.getLength());
            //node.show();
            //node.print();
            //System.out.println("------------------------ " + Arrays.toString(array));
            Node n1 = new Node(5,5,array);
            //events.get(row).scanShow();
            //events.get(row).write(node);
            events.get(row).write(n1);
            //events.get(row).scanShow();
        }
    }
    
    public void run(DataSource source){
        dataFrame.reset();
        timeProcessing = 0L;
        countProcessing = 0L;
        Event ev = new Event(200*1024);
        List<Event> eList = Arrays.asList(new Event());
        while(source.hasNext()==true){
            //source.nextFrame(sourceFrame);
            //System.out.println("event capacity = " + ev.getEventBuffer().capacity());
            source.next(ev);
            
            dataFrame.getList().add(ev.copy());

            long then = System.currentTimeMillis();
            if(dataFrame.getList().size()>batchSize){
                countProcessing += dataFrame.getList().size();
                INDArray[]  inputs = Level3Utils.createData(dataFrame.getList());
                INDArray[] outputs = this.processor.getOutput(inputs);
                System.out.printf(" progress = %d , time %d msec, frequency = %8.2f Hz\n",
                        countProcessing, timeProcessing, ((double) countProcessing*1000.0)/timeProcessing );
                writeOutput(outputs,dataFrame.getList());
                //System.out.println(inputs[0]);
                //System.out.println(outputs[0]);
                for(Event e: dataFrame.getList()) writer.addEvent(e);
                dataFrame.reset();
            }
            long now = System.currentTimeMillis();
            timeProcessing += (now - then);
            
        }
        System.out.printf("processed %d, in time = %d msec\n",countProcessing, timeProcessing);
        writer.close();
    }
    
    public static void main(String[] args){
        
        OptionParser opt = new OptionParser("level3-process");
        
        opt.addRequired("-n", "network file name")
                .addOption("-o", "", "output file name");
        
        opt.parse(args);
        
        Level3ProcessFile l3f = new Level3ProcessFile();
        
        //l3f.networkFile = "level3_model_0b_850_epochs.network_0b.network";
        l3f.networkFile = opt.getOption("-n").stringValue();//"etc/networks/network-level3-default.network";
        
        l3f.batchSize = 4096;
        l3f.maxEvents = 50;
        l3f.initialize();
        //String file = "/Users/gavalian/Work/Software/project-10.7/distribution/caos/coda/decoder/output.h5";
        //String file = "rec_clas_005442.evio.00050-00054.hipo_daq.h5";
        String file = opt.getInputList().get(0);

        HipoReader r = new HipoReader();
        r.setDebugMode(0);
        r.open(file);
        l3f.run(r);
    }
}
