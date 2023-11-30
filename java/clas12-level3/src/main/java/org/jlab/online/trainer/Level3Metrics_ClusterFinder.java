/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.trainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.evaluation.regression.RegressionEvaluation;

import twig.data.H1F;
import twig.graphics.TGCanvas;
import twig.data.GraphErrors;

/**
 *
 * @author tyson
 */
public class Level3Metrics_ClusterFinder {
    
    public Level3Metrics_ClusterFinder(long NEvents,INDArray predictions, INDArray Labels,Boolean makePlots){

		//System.out.print(Labels);

		RegressionEvaluation eval = new RegressionEvaluation(Labels.shape()[1]);
		eval.eval(Labels, predictions);
		System.out.printf("Test Average MAE: %f, MSE: %f\n",eval.averageMeanAbsoluteError(),eval.averageMeanSquaredError());
		System.out.printf("Test Average RMSE: %f, relativeSE: %f\n",eval.averagerootMeanSquaredError(),eval.averagerelativeSquaredError());
		if(makePlots){
        	PlotMetricsVsThreshold(NEvents,predictions,Labels);
			//Level3ClusterFinder_Simulation.applyThreshold(predictions, 0.1);
			for(int i=0;i<10;i++){
				plotExamples(predictions, Labels, i);
			}
		}
    }

	public static void plotExamples(INDArray predictions,INDArray Labels,int ex){
		TGCanvas c = new TGCanvas();
		c.setTitle("ECAL");

		H1F hTruth = new H1F("Truth", 108, 0, 108);
		hTruth.attr().setLineColor(2);
		hTruth.attr().setLineWidth(3);
		hTruth.attr().setTitleX("ECAL");
		hTruth.attr().setTitle("Truth");

		H1F hPred = new H1F("Predicted", 108, 0, 108);
		hPred.attr().setLineColor(5);
		hPred.attr().setLineWidth(3);
		hPred.attr().setTitleX("ECAL");
		hPred.attr().setTitle("Predicted");

		for (int j=0;j<predictions.shape()[1];j++){
			if(Labels.getFloat(ex, j)>0){
				hTruth.fill(j,Labels.getFloat(ex, j));
			}
			if(predictions.getFloat(ex, j)>0){
				hPred.fill(j,predictions.getFloat(ex, j));
			}
		}
		c.draw(hTruth).draw(hPred,"same");
		c.region().showLegend(0.05, 0.95);

	}

	public static void PlotMetricsVsThreshold(long NEvents,INDArray predictions,INDArray Labels){

		GraphErrors gMSE = new GraphErrors();
		gMSE.attr().setMarkerColor(2);
		gMSE.attr().setMarkerSize(10);
		gMSE.attr().setTitle("MSE");
		gMSE.attr().setTitleX("Threshold");
		gMSE.attr().setTitleY("Metrics");
		GraphErrors gMAE = new GraphErrors();
		gMAE.attr().setMarkerColor(5);
		gMAE.attr().setMarkerSize(10);
		gMAE.attr().setTitle("MAE");
		gMAE.attr().setTitleX("Threshold");
		gMAE.attr().setTitleY("Metrics");

		// Loop over threshold on the response
		for (double Th = 0.001; Th < 0.1; Th += 0.001) {
			INDArray predictionsThreshed=predictions.dup();
			Level3ClusterFinder_Simulation.applyThreshold(predictionsThreshed,Th);
			RegressionEvaluation eval = new RegressionEvaluation(Labels.shape()[1]);
			eval.eval(Labels, predictionsThreshed);
			gMAE.addPoint(Th, eval.averageMeanAbsoluteError(), 0, 0);
			gMSE.addPoint(Th, eval.averageMeanSquaredError(), 0, 0);
		} // Increment threshold on response

		TGCanvas c = new TGCanvas();
		c.setTitle("Metrics vs Threshold");
		c.draw(gMAE).draw(gMSE, "same");
		c.region().showLegend(0.05, 0.95);

	}


    

}
