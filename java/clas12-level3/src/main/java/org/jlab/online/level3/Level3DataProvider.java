/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Interface.java to edit this template
 */
package org.jlab.online.level3;

import j4np.hipo5.data.Event;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author gavalian
 */
public interface Level3DataProvider {
    public INDArray getData(List<Event> events);
}
