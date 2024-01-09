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
        	//PlotMetricsVsThreshold(NEvents,predictions,Labels);
			
			for(int i=0;i<5;i++){
				plotExamples(predictions, Labels, i);
			}
		}

		Level3ClusterFinder_Simulation.applyThreshold(predictions, 0.1);
		isEmpty(NEvents, predictions);
    }

	public static void isEmpty(long NEvents,INDArray predictions){
		H1F hNonNull = new H1F("Dist", 10,0,10);
		hNonNull.attr().setLineColor(2);
		hNonNull.attr().setLineWidth(3);
		hNonNull.attr().setTitleX("Number of Non-Zero Strips in Prediction");
		hNonNull.attr().setTitle("Number of Non-Zero Strips in Prediction");
		for (int i=0;i<NEvents;i++){
			int nonnull=0;
            for (int j = 0; j < 108; j++) {
                if (predictions.getFloat(i,j) > 0) {
                    nonnull++;
                }
            }
			hNonNull.fill(nonnull);
		}
		TGCanvas c = new TGCanvas();
		c.setTitle("Number of Non-Zero Strips in Prediction");
		c.draw(hNonNull,"");
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
		int nU=0,nV=0,nW=0;
		double nU_um=0,nV_um=0,nW_um=0;
		for (int i=0;i<NEvents;i++){
			List<Integer> CM_U=new ArrayList<>();
			List<Integer> CM_V=new ArrayList<>();
			List<Integer> CM_W=new ArrayList<>();
			List<Integer> CM_U_pred=new ArrayList<>();
			List<Integer> CM_V_pred=new ArrayList<>();
			List<Integer> CM_W_pred=new ArrayList<>();
			int CM_U_ind=-1,CM_V_ind=-1,CM_W_ind=-1;
			double ADC_max_U_pred=0,ADC_max_V_pred=0,ADC_max_W_pred=0;
			Boolean newCL_U=false,newCL_V=false,newCL_W=false;

			for (int j=0;j<36;j++){
				if(newCL_U==true){
					ADC_max_U_pred=0;
					if(CM_U_ind!=-1){CM_U_pred.add(CM_U_ind);}
					CM_U_ind=-1;
					newCL_U=false;
				} 
				if(newCL_V==true){
					ADC_max_V_pred=0;
					if(CM_V_ind!=-1){CM_V_pred.add(CM_V_ind);}
					CM_V_ind=-1;
					newCL_V=false;
				} 
				if(newCL_W==true){
					ADC_max_W_pred=0;
					if(CM_W_ind!=-1){CM_W_pred.add(CM_W_ind);}
					CM_W_ind=-1;
					newCL_W=false;
				} 
				int index_u=j,index_v=j+36,index_w=j+72;
				if(Labels.getFloat(i,index_u)>0){
					CM_U.add(index_u);
				}
				if(Labels.getFloat(i,index_v)>0){
					CM_V.add(index_v);
				}
				if(Labels.getFloat(i,index_w)>0){
					CM_W.add(index_w);
				}
				if(predictions.getFloat(i,index_u)==0){
					newCL_U=true;
				}
				if(predictions.getFloat(i,index_u)>ADC_max_U_pred){
					ADC_max_U_pred=predictions.getFloat(i,index_u);
					CM_U_ind=index_u;
				}
				if(predictions.getFloat(i,index_v)==0){
					newCL_V=true;
				}
				if(predictions.getFloat(i,index_v)>ADC_max_V_pred){
					ADC_max_V_pred=predictions.getFloat(i,index_v);
					CM_V_ind=index_v;
				}
				if(predictions.getFloat(i,index_w)==0){
					newCL_W=true;
				}
				if(predictions.getFloat(i,index_w)>ADC_max_W_pred){
					ADC_max_W_pred=predictions.getFloat(i,index_w);
					CM_W_ind=index_w;
				}
			}
			if(CM_U_ind!=-1){CM_U_pred.add(CM_U_ind);}
			if(CM_V_ind!=-1){CM_V_pred.add(CM_V_ind);}
			if(CM_W_ind!=-1){CM_W_pred.add(CM_W_ind);}
			if(CM_U.size()>0){
				nU+=CM_U.size();
				
				for(int CM_u : CM_U){
					int dist=99999;
					//System.out.printf("CM U %d\n", CM_u);
					for(int CM_u_p:CM_U_pred){
						//System.out.printf("CM U Pred %d\n",CM_u_p);
						//System.out.printf("Dist %d & CM_u-CM_u_p %d\n", dist,(CM_u-CM_u_p));
						if(Math.abs(CM_u-CM_u_p)<Math.abs(dist)){dist=CM_u-CM_u_p;}
						//System.out.printf("Dist %d\n", dist);
					}
					//System.out.println("\n");
					if(dist<36){//max dist as ecin only has 36 strips in each view
						Av_Dist_U+=dist;
						hDist_U.fill(dist);
						hDist.fill(dist);
					} else if( dist==99999){
						nU_um++;
					}
				}
				
			}
			if(CM_V.size()>0){
				nV+=CM_V.size();
				
				for(int CM_v : CM_V){
					int dist=99999;
					for(int CM_v_p:CM_V_pred){
						if(Math.abs(CM_v-CM_v_p)<Math.abs(dist)){dist=CM_v-CM_v_p;}
					}
					if(dist<36){//max dist as ecin only has 36 strips in each view
						Av_Dist_V+=dist;
						hDist_V.fill(dist);
						hDist.fill(dist);
					}else if( dist==99999){
						nV_um++;
					}
				}
				
			}
			if(CM_W.size()>0){
				nW+=CM_W.size();
				
				for(int CM_w : CM_W){
					int dist=99999;
					for(int CM_w_p:CM_W_pred){
						if(Math.abs(CM_w-CM_w_p)<Math.abs(dist)){dist=CM_w-CM_w_p;}
					}
					if(dist<36){//max dist as ecin only has 36 strips in each view
						Av_Dist_W+=dist;
						hDist_W.fill(dist);
						hDist.fill(dist);
					}else if( dist==99999){
						nW_um++;
					}
				}
				
			}
		}
		double Av_Dist=(Av_Dist_U+Av_Dist_V+Av_Dist_W)/(nU+nV+nW);
		Av_Dist_U=Av_Dist_U/nU;
		Av_Dist_W=Av_Dist_W/nV;
		Av_Dist_V=Av_Dist_V/nW;
		
		double nU_um_r=nU_um/nU;
		double nV_um_r=nV_um/nV;
		double nW_um_r=nW_um/nW;
		double um_r=(nU_um+nV_um+nW_um)/(nU+nV+nW);

		System.out.printf("Average Distance %f in U %f in V %f in W %f\n",Av_Dist,Av_Dist_U,Av_Dist_V,Av_Dist_W);
		System.out.printf("Fraction of unmatched clusters U %f V %f W %f total %f\n\n",nU_um_r,nV_um_r,nW_um_r,um_r);
		if(makePlots){
			TGCanvas c = new TGCanvas();
			c.setTitle("Distance Between Cluster Maxima");
			c.draw(hDist_U,"same").draw(hDist_V,"same").draw(hDist_W,"same");//draw(hDist).
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
