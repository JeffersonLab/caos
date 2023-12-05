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
		calcDist(NEvents, predictions, Labels, makePlots);
		if(makePlots){
        	PlotMetricsVsThreshold(NEvents,predictions,Labels);
			//Level3ClusterFinder_Simulation.applyThreshold(predictions, 0.1);
			for(int i=0;i<5;i++){
				plotExamples(predictions, Labels, i);
			}
		}
    }

	public static void calcDist(long NEvents,INDArray predictions, INDArray Labels,Boolean makePlots){
		H1F hDist = new H1F("Dist", 20,-10,10);
		hDist.attr().setLineColor(1);
		hDist.attr().setLineWidth(3);
		hDist.attr().setTitleX("Distance Between Cluster Maxima [strips]");
		hDist.attr().setTitle("U/V/W Average");

		H1F hDist_U = new H1F("Dist", 20,-10,10);
		hDist_U.attr().setLineColor(4);
		hDist_U.attr().setLineWidth(3);
		hDist_U.attr().setTitleX("Distance Between Cluster Maxima [strips]");
		hDist_U.attr().setTitle("U View");

		H1F hDist_V = new H1F("Dist", 20,-10,10);
		hDist_V.attr().setLineColor(3);
		hDist_V.attr().setLineWidth(3);
		hDist_V.attr().setTitleX("Distance Between Cluster Maxima [strips]");
		hDist_V.attr().setTitle("V View");

		H1F hDist_W = new H1F("Dist", 20,-10,10);
		hDist_W.attr().setLineColor(5);
		hDist_W.attr().setLineWidth(3);
		hDist_W.attr().setTitleX("Distance Between Cluster Maxima [strips]");
		hDist_W.attr().setTitle("W View");

		double Av_Dist_U=0,Av_Dist_V=0,Av_Dist_W=0;
		for (int i=0;i<NEvents;i++){
			double ADC_max_U=0,ADC_max_V=0,ADC_max_W=0;
			int CM_U=-1,CM_V=-1,CM_W=-1;
			double ADC_max_U_pred=0,ADC_max_V_pred=0,ADC_max_W_pred=0;
			int CM_U_pred=-1,CM_V_pred=-1,CM_W_pred=-1;

			for (int j=0;j<36;j++){
				int index_u=j,index_v=j+36,index_w=j+72;
				if(Labels.getFloat(i,index_u)>ADC_max_U){
					ADC_max_U=Labels.getFloat(i,index_u);
					CM_U=index_u;
				}
				if(Labels.getFloat(i,index_v)>ADC_max_V){
					ADC_max_V=Labels.getFloat(i,index_v);
					CM_V=index_v;
				}
				if(Labels.getFloat(i,index_w)>ADC_max_W){
					ADC_max_W=Labels.getFloat(i,index_w);
					CM_W=index_w;
				}
				if(predictions.getFloat(i,index_u)>ADC_max_U_pred){
					ADC_max_U_pred=predictions.getFloat(i,index_u);
					CM_U_pred=index_u;
				}
				if(predictions.getFloat(i,index_v)>ADC_max_V_pred){
					ADC_max_V_pred=predictions.getFloat(i,index_v);
					CM_V_pred=index_v;
				}
				if(predictions.getFloat(i,index_w)>ADC_max_W_pred){
					ADC_max_W_pred=predictions.getFloat(i,index_w);
					CM_W_pred=index_w;
				}
			}
			double Av_Dist_ev=0;
			int nCl=0;
			if(CM_U!=-1){
				double dist_U=(CM_U-CM_U_pred);
				Av_Dist_U+=dist_U;
				Av_Dist_ev+=dist_U;
				nCl++;
				hDist_U.fill(dist_U);
			}
			if(CM_V!=-1){
				double dist_V=(CM_V-CM_V_pred);
				Av_Dist_V+=dist_V;
				Av_Dist_ev+=dist_V;
				nCl++;
				hDist_V.fill(dist_V);
			}
			if(CM_W!=-1){
				double dist_W=(CM_W-CM_W_pred);
				Av_Dist_W+=dist_W;
				Av_Dist_ev+=dist_W;
				nCl++;
				hDist_W.fill(dist_W);
			}
			if(nCl!=0){hDist.fill(Av_Dist_ev/nCl);}
		}
		Av_Dist_U=Av_Dist_U/NEvents;
		Av_Dist_W=Av_Dist_W/NEvents;
		Av_Dist_V=Av_Dist_V/NEvents;
		double Av_Dist=(Av_Dist_U+Av_Dist_V+Av_Dist_W)/3;

		System.out.printf("Average Distance %f in U %f in V %f in W %f\n",Av_Dist,Av_Dist_U,Av_Dist_V,Av_Dist_W);
		if(makePlots){
			TGCanvas c = new TGCanvas();
			c.setTitle("Distance Between Cluster Maxima");
			c.draw(hDist).draw(hDist_U,"same").draw(hDist_V,"same").draw(hDist_W,"same");
			c.region().showLegend(0.05, 0.95);

		}

	}

	public static void plotExamples(INDArray predictions,INDArray Labels,int ex){
		TGCanvas c = new TGCanvas();
		c.setTitle("ECAL");

		H1F hTruth = new H1F("Truth", 109, 0, 109);
		hTruth.attr().setLineColor(2);
		hTruth.attr().setLineWidth(3);
		hTruth.attr().setTitleX("ECAL");
		hTruth.attr().setTitle("Truth");

		H1F hPred = new H1F("Predicted", 109, 0, 109);
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
