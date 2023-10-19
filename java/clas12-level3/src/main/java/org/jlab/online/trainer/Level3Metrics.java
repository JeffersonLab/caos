/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import twig.data.H1F;
import twig.graphics.TGCanvas;
import twig.data.GraphErrors;

/**
 *
 * @author tyson
 */
public class Level3Metrics {
    
    public Level3Metrics(long NEvents,INDArray predictions, INDArray Labels){
        PlotMetricsVsResponse(NEvents,predictions,Labels);
	PlotResponse(NEvents,predictions,Labels);
    }

    public static INDArray getMetrics(long NEvents,INDArray predictions, INDArray Labels, double RespTh) {
	INDArray metrics = Nd4j.zeros(3,1);
	double TP=0,FN=0,FP=0,TN=0;
	for(long i=0;i<NEvents;i+=1) {
	    if(Labels.getFloat(i,1)==1) {
		if(predictions.getFloat(i,1)>RespTh) {
		    TP++;
		} else {
		    FN++;
		}//Check model prediction
	    } else if(Labels.getFloat(i,1)==0) {
		if(predictions.getFloat(i,1)>RespTh) {
		    FP++;
		} else {
		    TN++;
		}//Check model prediction
	    }//Check true label
	}//loop over events
	double Acc=(TP+TN)/(TP+TN+FP+FN);
	double Pur=TP/(TP+FP);
	double Eff=TP/(TP+FN);
	metrics.putScalar(new int[] {0,0}, Acc);
	metrics.putScalar(new int[] {1,0}, Pur);
	metrics.putScalar(new int[] {2,0}, Eff);
	return metrics;
    }//End of getMetrics

    public static double PlotMetricsVsResponse(long NEvents,INDArray predictions, INDArray Labels) {
	GraphErrors gAcc= new GraphErrors();
	gAcc.attr().setMarkerColor(4);
	gAcc.attr().setMarkerSize(10);
	gAcc.attr().setTitle("Accuracy");
	gAcc.attr().setTitleX("Response");
	gAcc.attr().setTitleY("Metrics");
	GraphErrors gEff= new GraphErrors();
	gEff.attr().setMarkerColor(2);
	gEff.attr().setMarkerSize(10);
	gEff.attr().setTitle("Efficiency");
	gEff.attr().setTitleX("Response");
	gEff.attr().setTitleY("Metrics");
	GraphErrors gPur= new GraphErrors();
	gPur.attr().setMarkerColor(5);
	gPur.attr().setMarkerSize(10);
	gPur.attr().setTitle("Purity");
	gPur.attr().setTitleX("Response");
	gPur.attr().setTitleY("Metrics");
	double bestRespTh=0;
	double bestPuratEff0p995=0;
		
	//Loop over threshold on the response
	for(double RespTh=0.01; RespTh<0.99;RespTh+=0.01) {
	    INDArray metrics =getMetrics(NEvents, predictions, Labels, RespTh);
	    double Acc=metrics.getFloat(0,0);
	    double Pur=metrics.getFloat(1,0);
	    double Eff=metrics.getFloat(2,0);
	    gAcc.addPoint(RespTh, Acc, 0, 0);
	    gPur.addPoint(RespTh, Pur, 0, 0);
	    gEff.addPoint(RespTh, Eff, 0, 0);
	    if(Eff>0.995) {
		if (Pur>bestPuratEff0p995) {
		    bestPuratEff0p995=Pur;
		    bestRespTh=RespTh;
		}
	    }
	}//Increment threshold on response
		
	System.out.format("%n Best Purity at Efficiency above 0.995: %.3f at a threshold on the response of %.3f %n%n",bestPuratEff0p995,bestRespTh);

	TGCanvas c = new TGCanvas();
	c.setTitle("Metrics vs Response");
	c.draw(gAcc).draw(gEff,"same").draw(gPur,"same");
		
	return bestRespTh;
    }//End of PlotMetricsVSResponse

    public static void PlotResponse(long NEvents,INDArray output, INDArray Labels) {
	H1F hRespPos = new H1F("Positive Sample", 100, 0, 1);
	hRespPos.attr().setLineColor(2);
	hRespPos.attr().setFillColor(2);
	hRespPos.attr().setLineWidth(3);
	hRespPos.attr().setTitleX("Response");
	H1F hRespNeg = new H1F("Negative Sample", 100, 0, 1);
	hRespNeg.attr().setLineColor(5);
	hRespNeg.attr().setLineWidth(3);
	hRespNeg.attr().setTitleX("Response");
	//Sort predictions into those made on the positive/or negative samples
	for(long i=0;i<NEvents;i+=1) {
	    if(Labels.getFloat(i,1)==1) {
		hRespPos.fill(output.getFloat(i,1));
	    } else if(Labels.getFloat(i,1)==0) {
		hRespNeg.fill(output.getFloat(i,1));
	    }
	}

	TGCanvas c = new TGCanvas();
	c.setTitle("Response");
	c.draw(hRespPos).draw(hRespNeg,"same");
		
    }//End of PlotResponse

}
