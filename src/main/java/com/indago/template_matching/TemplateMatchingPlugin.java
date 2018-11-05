package com.indago.template_matching;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

/**
 * Template Matching Plugin for Fiji/ImageJ2
 *
 * @author Mangal Prakash
 */

@Plugin( menuPath = "Plugins>Segmentation>Template Matching Segmentation", type = Command.class )

public class TemplateMatchingPlugin< T extends RealType< T > & NativeType< T > > implements Command {

	@Parameter
	DatasetIOService datasetIOService;

	@Parameter( label = "Image to load" )
	private File inputImage;

	@Parameter( label = "Template to load" )
	private File inputTemplate;

	@Parameter( style = "directory" )
	private File saveResultsDir;

	@Parameter( label = "Segmentation circle radius", persist = false, min = "1" )
	private int segCircleRad = 3;

	@Parameter( label = "Matching Threshold", persist = false, min = "0.1", max = "1.0", stepSize = "0.05" )
	private double threshold = 0.3;

	@Parameter
	StatusService statusService;
	
	@Parameter
	UIService uiService;

	@Parameter
	private Context context;

	public static void main( String[] args ) throws IOException {

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		ij.command().run( TemplateMatchingPlugin.class, true );
	}

	@Override
	public void run() {
		try {
			templateMatching(  );
		} catch ( Exception e ) {
			e.printStackTrace();
		}

	}

	private void templateMatching()
			throws Exception {
		
		Dataset imagefile = datasetIOService.open( inputImage.getAbsolutePath() );
		Dataset templateFile = datasetIOService.open( inputTemplate.getAbsolutePath() );
		File saveDir = saveResultsDir;
		int segRadius = segCircleRad;

		ImgPlus< T > imp = ( ImgPlus< T > ) imagefile.getImgPlus();
		ImgPlus< T > template = ( ImgPlus< T > ) templateFile.getImgPlus();
		double thresholdmatch = threshold;
		StatusService statusService = this.statusService;
		List< RandomAccessibleInterval< T > > trueSegmentations =
				new TemplateMatchingAlgorithm( context ).calculate( imp, template, segRadius, thresholdmatch, statusService );
		Utilities.saveImagesToDirectory( trueSegmentations, saveDir );

		for ( Object results : trueSegmentations ) {
			uiService.show( results );
		}
	}
}
