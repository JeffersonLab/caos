/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.data.h5;

import j4np.data.base.DataFrame;
import j4np.hipo5.data.Event;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

/**
 *
 * @author gavalian
 */
public class MonitorSourceEt  {
    
    Timer etTimer = null;
    DataFrame<Event> etEvents = new DataFrame<>();
    DataSourceEt etSource = null;
    DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    
    public MonitorSourceEt(String file, int interval){
        etSource = new DataSourceEt("localhost");
        etSource.open(file);
        this.initTimer(interval);
    }
    
    private void initTimer(int interval){
        TimerTask timerTask = new TimerTask() {
            @Override
            public void run() {
                onTimerEvent();
                /*for(int i = 0; i < canvasPads.size();i++){
                    System.out.println("PAD = " + i);
                    canvasPads.get(i).show();
                }*/
            }
        };
        etTimer = new Timer("EmbeddeCanvasTimer");
        etTimer.scheduleAtFixedRate(timerTask, 30, interval);
    }
    
    private void onTimerEvent(){
        Date date = new Date();
        try {
            etSource.nextFrame(etEvents);
            int size = 0;

            System.out.printf("%s: receive buffer with %d events\n",
                    dateFormat.format(date),etEvents.getCount());
        } catch (Exception e){
            System.out.printf("%s: error receiving events from et ring.....\n",
                    dateFormat.format(date));
        }
    }
    
}
