package org.jlab.online.trainer;

import j4np.hipo5.data.Bank;

public class Level3Particle {

    int PID=0;

    int PIndex=0;
    int Sector=0;
    int Charge=0;

    double Px=0;
    double Py=0;
    double Pz=0;

    double P=0;
    double Theta=0;
    double Phi=0;

    double Nphe=0;

    int Cal_index=-1;

    double PCAL_energy=0;
    double ECIN_energy=0;
    double ECOUT_energy=0;
    double PCALLU=0;
    double PCALLV=0;
    double PCALLW=0;

    double ECINLU=0;
    double ECINLV=0;
    double ECINLW=0;

    double ECOUTLU=0;
    double ECOUTLV=0;
    double ECOUTLW=0;

    double PCAL_scale=6.5;
    double ECAL_scale=8.0;

    double PCal_UMax_cut = 80.;
    double PCal_UMin_cut = 3.; // For oubendings this doesn't matter much;
    double PCal_VMax_cut = 73.;
    double PCal_VMin_cut = 3.;
    double PCal_WMax_cut = 73.;
    double PCal_WMin_cut = 3.;

    double ECin_UMax_cut = 35.;
    double ECin_UMin_cut = 2.5; // For oubendings this doesn't matter much;
    double ECin_VMax_cut = 34.;
    double ECin_VMin_cut = 3.;
    double ECin_WMax_cut = 34.;
    double ECin_WMin_cut = 3.;

    double ECout_UMax_cut = 34.;
    double ECout_UMin_cut = 2; // For oubendings this doesn't matter much;
    double ECout_VMax_cut = 34;
    double ECout_VMin_cut = 3.;
    double ECout_WMax_cut = 34.;
    double ECout_WMin_cut = 3.;

    public Level3Particle(){

    }

    public void find_sector_track(Bank TrackBank){
        for (int k = 0; k < TrackBank.getRows(); k++) {
            int pindex = TrackBank.getInt("pindex", k);
            int sectorTrk = TrackBank.getInt("sector", k);
            if(pindex==PIndex){Sector=sectorTrk;}
        }
    }

    public void read_HTCC_bank(Bank HTCCBank){
        for (int k = 0; k < HTCCBank.getRows(); k++) {
            int pindex = HTCCBank.getInt("pindex", k);
            double nphe = HTCCBank.getFloat("nphe", k);
            if(pindex==PIndex){Nphe=nphe;}
        }
    }

    public Boolean check_FID_Cal_Clusters(Bank ECAL_Bank){
        Boolean Fid=true;
        //ECAL_Bank.show();
        //System.out.printf("index %d sector %d\n",PIndex,Sector);
        for (int k = 0; k < ECAL_Bank.getRows(); k++) {
            int sector=ECAL_Bank.getInt("sector", k);
            int layer=ECAL_Bank.getInt("layer",k);
            double u=(double) ECAL_Bank.getInt("coordU",k);
            double v=(double) ECAL_Bank.getInt("coordV",k);
            double w=(double) ECAL_Bank.getInt("coordW",k);
            double energy=ECAL_Bank.getFloat("energy",k);
            
            if(layer==1 && k==Cal_index){
                u=u/PCAL_scale;
                v=v/PCAL_scale;
                w=w/PCAL_scale;
                /*System.out.printf("layer %d, sector %d, u %f,v %f,w %f,energy %f\n",layer,sector,u,v,w,energy);
                System.out.printf("Sector %d, energy PCAL %f, ECIN %f , ECOUT %f\n",Sector,PCAL_energy,ECIN_energy,ECOUT_energy);*/
                if(u>PCal_UMax_cut||u<PCal_UMin_cut){Fid=false;}
                if(v>PCal_VMax_cut||v<PCal_VMin_cut){Fid=false;}
                if(w>PCal_WMax_cut||w<PCal_WMin_cut){Fid=false;}
            } else if(layer==4 && k==Cal_index){
                u=u/ECAL_scale;
                v=v/ECAL_scale;
                w=w/ECAL_scale;
                /*System.out.printf("layer %d, sector %d, u %f,v %f,w %f,energy %f\n",layer,sector,u,v,w,energy);
                System.out.printf("Sector %d, energy PCAL %f, ECIN %f , ECOUT %f\n",Sector,PCAL_energy,ECIN_energy,ECOUT_energy);*/
                if(u>ECin_UMax_cut||u<ECin_UMin_cut){Fid=false;}
                if(v>ECin_VMax_cut||v<ECin_VMin_cut){Fid=false;}
                if(w>ECin_WMax_cut||w<ECin_WMin_cut){Fid=false;}
            } else if(layer==7 && k==Cal_index){
                u=u/ECAL_scale;
                v=v/ECAL_scale;
                w=w/ECAL_scale;
                /*System.out.printf("layer %d, sector %d, u %f,v %f,w %f,energy %f\n",layer,sector,u,v,w,energy);
                System.out.printf("Sector %d, energy PCAL %f, ECIN %f , ECOUT %f\n",Sector,PCAL_energy,ECIN_energy,ECOUT_energy);*/
                if(u>ECout_UMax_cut||u<ECout_UMin_cut){Fid=false;}
                if(v>ECout_VMax_cut||v<ECout_VMin_cut){Fid=false;}
                if(w>ECout_WMax_cut||w<ECout_WMin_cut){Fid=false;}
            }
            
        }
        return Fid;
    }

    public Boolean check_Energy_Dep_Cut(){
        Boolean pass=false;
        double tot_e=PCAL_energy+ECIN_energy+ECOUT_energy;
        //System.out.printf("energy P %f i %f o %f t %f\n",PCAL_energy,ECIN_energy,ECOUT_energy,tot_e);
        if(tot_e>0.25&&PCAL_energy>0.06){
            pass=true;
        }
        return pass;
    }

    public Boolean check_SF_cut(){
        Boolean pass=false;
        double tot_e=PCAL_energy+ECIN_energy+ECOUT_energy;
        double sf=tot_e/P;
        if(sf>0.2){pass=true;}
        return pass;
    }

    public void read_Cal_Bank(Bank ECAL_Bank){
        //ECAL_Bank.show();
        for (int k = 0; k < ECAL_Bank.getRows(); k++) {
            int pindex = ECAL_Bank.getInt("pindex", k);
            int index = ECAL_Bank.getInt("index", k);
            double energy = ECAL_Bank.getFloat("energy", k);
            int sector = ECAL_Bank.getInt("sector", k);
            int layer=ECAL_Bank.getInt("layer",k);
            double lu=ECAL_Bank.getFloat("lu",k);
            double lv=ECAL_Bank.getFloat("lv",k);
            double lw=ECAL_Bank.getFloat("lw",k);
            if (pindex == PIndex) {
                Cal_index=index;
                if(layer==1){
                    PCALLU=lu;
                    PCALLV=lv;
                    PCALLW=lw;
                    PCAL_energy=energy;
                } else if(layer==4){
                    ECINLU=lu;
                    ECINLV=lv;
                    ECINLW=lw;
                    ECIN_energy=energy;
                } else if(layer==7){
                    ECOUTLU=lu;
                    ECOUTLV=lv;
                    ECOUTLW=lw;
                    ECOUT_energy=energy;
                }
            }
        }

    }

    public void read_Particle_Bank(int pindex, Bank PartBank) {
        int pid = PartBank.getInt("pid", pindex);
        int status = PartBank.getInt("status", pindex);
        int charge = PartBank.getInt("charge", pindex);
        double px = PartBank.getFloat("px", pindex);
        double py = PartBank.getFloat("py", pindex);
        double pz = PartBank.getFloat("pz", pindex);
        double[] pthetaphi = new double[3];
        pthetaphi[0] = Math.sqrt(px * px + py * py + pz * pz);
        pthetaphi[1] = Math.acos(pz / pthetaphi[0]);// Math.atan2(Math.sqrt(px*px+py*py),pz);
        pthetaphi[2] = Math.atan2(py, px);
        if (Math.abs(status) >= 2000 && Math.abs(status) < 4000) {
            PID = pid;
            Charge = charge;
            Px = px;
            Py = py;
            Pz = pz;
            P = pthetaphi[0];
            Theta = pthetaphi[1];
            Phi = pthetaphi[2];
            PIndex = pindex;
        }
    }

}
