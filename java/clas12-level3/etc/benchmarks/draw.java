import java.awt.Font;
public class Draw {

    public static GraphErrors getGraph(String file, int size, int color){
	List<DataVector> v = DataIO.load(file,"\\s+",0,2);
	GraphErrors g = new GraphErrors("graph",v.get(0),v.get(1));
	g.attr().set(String.format("mc=%d,lc=%d,ms=%d,lw=2",color,color,size));
	g.attr().setTitleX("CPU cores");
	g.attr().setTitleY("Inference Rate (Hz)");
	return g;
    }

    public static void draw(){
	TStyle.getInstance().setDefaultAxisLabelFont(new Font("Palatino",Font.BOLD,24));
	TStyle.getInstance().setDefaultAxisTitleFont(new Font("Palatino",Font.BOLD,28));
	TStyle.getInstance().setDefaultPaveTextFont(new Font("Palatino",Font.BOLD,28));
	String[] files = new String[]{
	    "results/ahmdal_512.txt",
	    "results/ahmdal_1024.txt",
	    "results/ahmdal_2048.txt",
	    "results/ahmdal_4096.txt",
	    "results/ahmdal_8192.txt",
	};

	String[] labels = new String[]{"512","1024","2048","4096","8192"};
	TGCanvas c = new TGCanvas(800,900);
	c.region().getAxisFrame().setBackgroundColor(230,230,230);
	c.view().left(120);
	for(int l = 0; l < files.length; l++){
	    GraphErrors g = Draw.getGraph(files[l],12,l+2);
	    g.attr().setLegend("batch size " + labels[l]);
	    c.draw(g,"PLsame");
	}

	c.region().showLegend(0.05,0.98);
    }
}
