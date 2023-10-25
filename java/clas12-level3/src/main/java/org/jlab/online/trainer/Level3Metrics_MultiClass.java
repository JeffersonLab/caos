/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.evaluation.classification.Evaluation;

import twig.data.H1F;
import twig.graphics.TGCanvas;
import twig.data.GraphErrors;

/**
 *
 * @author tyson
 */
public class Level3Metrics_MultiClass {
    
    public Level3Metrics_MultiClass(long NEvents,INDArray predictions, INDArray Labels,List<Integer> tags,Boolean makePlots){

		Evaluation eval = new Evaluation(tags.size());
		eval.eval(Labels, predictions);
		System.out.println(eval.stats());
		System.out.println(eval.confusionToString());
		if(makePlots){
        	PlotMetricsVsResponse(NEvents,predictions,Labels,tags);
			PlotResponse(NEvents,predictions,Labels,tags);
		}
    }

    public static INDArray getMetrics(long NEvents, INDArray predictions, INDArray Labels, double RespTh,
			List<Integer> tags) {

		int el_tag_index=findIndexTagN(tags, 1);
		INDArray metrics = Nd4j.zeros(2, 1);

		//System.out.printf("\n Electron tag at index: %d \n\n",el_tag_index);
		
		double TP = 0, FN = 0, FP = 0, TN = 0;
		for (long i = 0; i < NEvents; i += 1) {
			if (Labels.getFloat(i, el_tag_index) == 1) {
				if (predictions.getFloat(i, el_tag_index) > RespTh) {
					TP++;
				} else {
					FN++;
				} // Check model prediction
			} else if (Labels.getFloat(i, el_tag_index) == 0) {
				if (predictions.getFloat(i, el_tag_index) > RespTh) {
					FP++;
				} else {
					TN++;
				} // Check model prediction
			} // Check true label
		} // loop over events

		double Pur = TP/(TP+FP);
		double Eff = TP/(TP+TN);
		metrics.putScalar(new int[] { 0, 0 }, Pur);
		metrics.putScalar(new int[] { 1, 0 }, Eff);
		return metrics;
	}// End of getMetrics

	public static void PlotMetricsVsResponse(long NEvents, INDArray predictions, INDArray Labels,List<Integer> tags) {

		GraphErrors gEff = new GraphErrors();
		gEff.attr().setMarkerColor(2);
		gEff.attr().setMarkerSize(10);
		gEff.attr().setTitle("Efficiency");
		gEff.attr().setTitleX("Response");
		gEff.attr().setTitleY("Metrics");
		GraphErrors gPur = new GraphErrors();
		gPur.attr().setMarkerColor(5);
		gPur.attr().setMarkerSize(10);
		gPur.attr().setTitle("Purity");
		gPur.attr().setTitleX("Response");
		gPur.attr().setTitleY("Metrics");
		double bestRespTh = 0;
		double bestPuratEff0p995 = 0;

		// Loop over threshold on the response
		for (double RespTh = 0.01; RespTh < 0.99; RespTh += 0.01) {
			INDArray metrics = getMetrics(NEvents, predictions, Labels, RespTh,tags);
			double Pur = metrics.getFloat(0, 0);
			double Eff = metrics.getFloat(1, 0);
			gPur.addPoint(RespTh, Pur, 0, 0);
			gEff.addPoint(RespTh, Eff, 0, 0);
			if (Eff > 0.995) {
				if (Pur > bestPuratEff0p995) {
					bestPuratEff0p995 = Pur;
					bestRespTh = RespTh;
				}
			}
		} // Increment threshold on response

		System.out.format("%n Best Purity at Efficiency above 0.995: %.3f at a threshold on the response of %.3f %n%n",
				bestPuratEff0p995, bestRespTh);

		TGCanvas c = new TGCanvas();
		c.setTitle("Electron Metrics vs Response");
		c.draw(gEff).draw(gPur, "same");

	}// End of PlotMetricsVSResponse

	public static int findIndexTagN(List<Integer> tags,int tagN){
		int index=-1;
		for (int i=0;i<tags.size();i++) {
			if (tags.get(i)==tagN){index=i;}
		}
		return index;
	}

    public static void PlotResponse(long NEvents, INDArray output, INDArray Labels, List<Integer> tags) {

		int index_tag1=findIndexTagN(tags,1);

		TGCanvas c = new TGCanvas();
		c.setTitle("Response");

		int tag_counter=0;
		//loop over tags/classes
		for (int tag : tags) {
			H1F hResp = new H1F("Positive Sample", 100, 0, 1);
			hResp.attr().setLineColor(tag+1);//tags start at 1
			if (tag == 1) {
				hResp.attr().setFillColor(2);
			}
			hResp.attr().setLineWidth(3);
			hResp.attr().setTitleX("Response");
			hResp.attr().setTitle("Tag "+String.valueOf(tag));

			//System.out.printf("\n Plot response in class %d of tag %d \n\n",index_tag1,tag);

			// Sort predictions into those made on a given tag/class
			for (long i = 0; i < NEvents; i += 1) {
				//check if this event belongs to the tag/class
				if (Labels.getFloat(i, tag_counter) == 1) {
					//always fill with response for class 1
					//we want to see how hard a class is to distinguish
					//from class 1
					hResp.fill(output.getFloat(i, index_tag1));
				} 
			}

			if(tag_counter==0){
				c.draw(hResp);
			} else{
				c.draw(hResp,"same");
			}
			tag_counter++;

		}
		
    }//End of PlotResponse

}
