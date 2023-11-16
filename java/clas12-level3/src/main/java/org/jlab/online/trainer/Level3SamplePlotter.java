/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import j4np.data.base.DataFrame;
import j4np.hipo5.data.Bank;
import j4np.hipo5.data.CompositeNode;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jlab.online.level3.Level3Utils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.List;
import java.io.File;
import java.io.IOException;
import java.lang.Math;

import twig.data.GraphErrors;
import twig.data.H1F;
import twig.data.H2F;
import twig.graphics.TGCanvas;

/**
 *
 * @author tyson
 */
public class Level3SamplePlotter {

    

    public Level3SamplePlotter() {

    }

    public static H1F makeNiceHist(String title, String axis, int Color, int nBins, double low, double high) {
        H1F h = new H1F(title, nBins, low, high);
        h.attr().setLineColor(Color);
        //h.attr().setFillColor(2);
        h.attr().setLineWidth(3);
        h.attr().setTitleX(axis);
        return h;
    }

    public static H2F makeNice2DHist(String title, String axis, int nBins, double low, double high,String axis2,int nBins2, double low2, double high2) {
        H2F h = new H2F(title, nBins, low, high, nBins2, low2, high2);
        h.attr().setTitleX(axis);
        h.attr().setTitleY(axis2);
        return h;
    }

    
   
    public static void plot(String file,int max,double beamE){

        HipoReader r = new HipoReader(file);
        Event e = new Event();

        int nMax = max;
        if (r.entries() < max)
            nMax = r.entries();

        
        
        Bank[] banks = r.getBanks("DC::tdc","ECAL::adc","RUN::config");
        Bank[]  dsts = r.getBanks("REC::Particle","REC::Track","REC::Calorimeter","REC::Cherenkov");
        
        H1F elP=makeNiceHist("e-","Momentum [GeV]",2,110,0,11);
        H1F poP=makeNiceHist("e+","Momentum [GeV]",3,110,0,11);
        H1F negP=makeNiceHist("Negative","Momentum [GeV]",5,110,0,11);
        H1F posP=makeNiceHist("Positive","Momentum [GeV]",6,110,0,11);

        H2F elPTheta=makeNice2DHist("e-","e- Momentum [GeV]",110,0,11,"e- Theta [Deg]",100,0,40);
        H2F poPTheta=makeNice2DHist("e+","e+ Momentum [GeV]",110,0,11,"e+ Theta [Deg]",100,0,40);
        H2F posPTheta=makeNice2DHist("Positive","Positives Momentum [GeV]",110,0,11,"Positives Theta [Deg]",100,0,40);
        H2F negPTheta=makeNice2DHist("Negative","Negatives Momentum [GeV]",110,0,11,"Negatives Theta [Deg]",100,0,40);
        
        int counter=0,nEls=0,nPosi=0,nPos=0,nNeg=0;
        while(r.hasNext() && counter<nMax){
            
            r.nextEvent(e);
            e.read(banks);
            e.read(dsts);

            long bits = banks[2].getLong("trigger", 0);
            int[] trigger = Level3Converter_MultiClass.convertTriggerLong(bits);
            //System.out.println(Arrays.toString(trigger));

            ////get part pIndex and p
            for (int i = 0; i < dsts[0].getRows(); i++) {
                int pid = dsts[0].getInt("pid", i);
                int status = dsts[0].getInt("status", i);
                int charge = dsts[0].getInt("charge", i);
                
                if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
                    double[] pthetaphi=Level3Converter_MultiClass.calcPThetaPhi(dsts[0], i);
                    if (pid == 11) {
                        nEls++;
                        elP.fill(pthetaphi[0]);
                        elPTheta.fill(pthetaphi[0],pthetaphi[1]*(180.0/Math.PI));
                    } else if(pid==-11){
                        nPosi++;
                        poP.fill(pthetaphi[0]);
                        poPTheta.fill(pthetaphi[0],pthetaphi[1]*(180.0/Math.PI));
                    }
                        
                    if(charge<0){
                        nNeg++;
                        negP.fill(pthetaphi[0]);
                        negPTheta.fill(pthetaphi[0],pthetaphi[1]*(180.0/Math.PI));
                    } else if (charge>0){
                        nPos++;
                        posP.fill(pthetaphi[0]);
                        posPTheta.fill(pthetaphi[0],pthetaphi[1]*(180.0/Math.PI));
                    }
                }
            }

        }
        //System.out.print(OUTArray);
        //System.out.print(DCArray);
        //System.out.print(ECArray);
        System.out.printf("counter %d, nEl %d, nPosi %d, nNeg %d, nPos %d\n\n",counter,nEls,nPosi,nNeg,nPos);

        TGCanvas cP = new TGCanvas();
        cP.setTitle("Momentum");
        cP.draw(elP).draw(poP,"same").draw(posP,"same").draw(negP,"same");
        cP.region().showLegend(0.7, 0.95);

        TGCanvas cPThel = new TGCanvas();
        cPThel.setTitle("e- Momentum v Theta");
        cPThel.draw(elPTheta);

        TGCanvas cPThpo = new TGCanvas();
        cPThpo.setTitle("e+ Momentum v Theta");
        cPThpo.draw(poPTheta);

        TGCanvas cPThpos = new TGCanvas();
        cPThpos.setTitle("Pos Momentum v Theta");
        cPThpos.draw(posPTheta);

        TGCanvas cPThneg = new TGCanvas();
        cPThneg.setTitle("Neg Momentum v Theta");
        cPThneg.draw(negPTheta);
    }

    
    
    public static void main(String[] args){        
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018437_AI/rec_clas_018437.evio.00005-00009.hipo";//_AI
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018331_AI/rec_clas_018331.evio.00105-00109.hipo";
        //String file2="/Users/tyson/data_repo/trigger_data/rgd/018326/run_018326_2.h5";
        String file2="/Users/tyson/data_repo/trigger_data/rgd/018777/run_018777.h5";

        //String file2="/Users/tyson/data_repo/trigger_data/rga/rec_clas_005197.evio.00005-00009.hipo";

        

        Level3SamplePlotter P=new Level3SamplePlotter();

        
        Level3SamplePlotter.plot(file2,600000,10.547);
        
    }
}
