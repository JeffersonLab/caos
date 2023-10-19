/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package org.jlab.online.level3;

import j4np.hipo5.data.Event;
import j4np.hipo5.data.Node;
import j4np.hipo5.io.HipoReader;
import java.util.Arrays;
import twig.data.H1F;
import twig.graphics.TGCanvas;

/**
 *
 * @author gavalian
 */
public class Level3Validate {
    public static void main(String[] args){
        HipoReader r = new HipoReader("output_level3_dict.h5");
        
        Event e = new Event();
        H1F h = new H1F("h",120,0,1);
        TGCanvas c = new TGCanvas();
        
        c.draw(h);
        while(r.hasNext()){
            r.next(e);
            Node at = e.read(5, 5);
            System.out.println(Arrays.toString(at.getFloat()));
            if(at.getDataSize()>0) h.fill(at.getFloat(0));
        }
    }
}
