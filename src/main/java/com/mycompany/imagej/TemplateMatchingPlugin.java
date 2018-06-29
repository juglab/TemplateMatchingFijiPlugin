package com.mycompany.imagej;

import static org.bytedeco.javacpp.opencv_imgproc.TM_CCOEFF_NORMED;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;

import java.io.IOException;
import java.util.ArrayList;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatExpr;
import org.scijava.command.Command;

import ij.ImagePlus;
import ijopencv.ij.ImagePlusMatConverter;
import ijopencv.opencv.MatImagePlusConverter;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class TemplateMatchingPlugin implements Command {

	private static final double pi = 3.14;
	private static final String SetZero = null;

	public static void main( String[] args ) throws IOException {
		TemplateMatchingPlugin instanceOfMine = new TemplateMatchingPlugin();
		instanceOfMine.run();
	}

	///let's see I am creating a new branch
	@Override
	public void run() {
		try {
			runThrowsException();
		} catch ( Exception e ) {
			e.printStackTrace();
		}
	}


	private < T extends RealType< T > > void runThrowsException() throws Exception {
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		String imagePathName = "/Users/prakash/Desktop/image_time#00000.tif";
		Dataset imagefile = ij.scifio().datasetIO().open( imagePathName );
		ImgPlus< T > img = ( ImgPlus< T > ) imagefile.getImgPlus();

		//Gaussian Smoothing of Image

		double[] sigmas = { 1.5, 1.5 };
		RandomAccessibleInterval< FloatType > imgSmooth = ij.op().filter().gauss( img, sigmas );
//		Img< DoubleType > doubles = ij.op().convert().float64( Views.iterable( imgSmooth ) );

		System.out.println( Util.getTypeFromInterval( imgSmooth ).getClass() );

		FloatType maxVal = new FloatType();
		ij.op().stats().max( maxVal, Views.iterable( imgSmooth ) );
		float inverse = 1.0f / maxVal.getRealFloat();

		//Normalizing the image
		LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );
		ij.ui().show( imgSmooth );

		//Load template 
		String templatePathName = "/Users/prakash/Desktop/raw_untemp1.tif";
		Dataset templatefile = ij.scifio().datasetIO().open( templatePathName );
		ImgPlus< T > template = ( ImgPlus< T > ) templatefile.getImgPlus();
		int tH = ( int ) template.dimension( 1 );
		int tW = ( int ) template.dimension( 0 );
		int padHFrom = tH / 2;
		int padWFrom = tW / 2;
		int padHTo = ( int ) ( img.dimension( 1 ) - padHFrom + 1 );
		int padWTo = ( int ) ( img.dimension( 0 ) - padWFrom + 1 );

		//Converters
		ImagePlusMatConverter ic = new ImagePlusMatConverter();
		ImagePlus wrappedImage = ImageJFunctions.wrap( img, "Image" );
		ImagePlus wrappedSmoothImage = ImageJFunctions.wrap( imgSmooth, "Smooth Image" );

		// Convert the image to OpenCV image
		opencv_core.Mat cvImage = ic.convert( wrappedImage, Mat.class );


		for ( int angle = 30; angle <= 30; angle = angle + 30 ) {
			
			//Rotate template
			RandomAccessibleInterval< T > templateRot = rotateAndShow( ij, template, angle );
			ImagePlus rot = ImageJFunctions.wrap( templateRot, "rotated" );

			// Convert the template to OpenCV image
			opencv_core.Mat cvTemplate = ic.convert( rot, Mat.class );
			opencv_core.Mat cvSmoothImage = ic.convert( wrappedSmoothImage, Mat.class );
			opencv_core.Mat tempResults = new opencv_core.Mat();

			matchTemplate( cvImage, cvTemplate, tempResults, TM_CCOEFF_NORMED );
			Mat results = cvImage.clone();

			//Setting all elements of results matrix to zero

			for ( int i = 0; i < results.arrayHeight(); i++ ) {
				for ( int j = 0; j < results.arrayWidth(); j++ ) {
					BytePointer zeroPtr = results.ptr( i, j );
					zeroPtr.fill( 0 );

				}
			}

			//Replacing the submatrix within results matrix with template matching results matrix

			for ( int i = padHFrom; i < padHTo; i++ ) {
				for ( int j = padWFrom; j < padWTo; j++ ) {
					BytePointer resultsPtr = results.ptr( i, j );
					BytePointer tempResultsPtr = tempResults.ptr( i - padHFrom, j - padWFrom );
					resultsPtr.fill( tempResultsPtr.get() );

				}
			}

			//Counteracting normalizedness of template matching by multiplying with the smoothed image intensity of raw image

			MatExpr prod = results.mul( cvSmoothImage );
			Mat resTimes = prod.asMat();

			MatImagePlusConverter mip = new MatImagePlusConverter();
			Img< FloatType > resTimesIntensity = ImageJFunctions.convertFloat( mip.convert( resTimes, ImagePlus.class ) );
			LoopBuilder.setImages( resTimesIntensity ).forEachPixel( a -> {
				float b = a.getRealFloat();
				if ( b > 0.3 ) {
					ArrayList hits = new ArrayList();
					hits.add( a.getIndex() );

				}
			} );

		}


	}

	private < T extends RealType< T > > RandomAccessibleInterval< T > rotateAndShow( final ImageJ ij, ImgPlus< T > img, int angle ) {
		long x = -img.dimension( 0 ) / 2;
		long y = -img.dimension( 1 ) / 2;
		AffineTransform2D transform = new AffineTransform2D();
		transform.translate( x, y );
		transform.rotate( angle );
		transform.translate( -x, -y );
		RealRandomAccessible< T > realview =
				RealViews.affineReal(
						( Views.interpolate( Views.extendBorder( img ), new NLinearInterpolatorFactory() ) ),
						transform );
		RandomAccessibleInterval< T > view = Views.interval( Views.raster( realview ), img );
		ij.ui().show( view );
		return(view);
	}

}

//Mat a = cvTemplate.clone();
//Mat bi = cvImage.clone();

//for ( int i = 0; i <= a.arrayHeight(); i++ ) {
//	for ( int j = 0; j <= a.arrayWidth(); j++ ) {
//		BytePointer aPtr = a.ptr( i, j );
//		aPtr.fill( 122 );
//	}
//}
//
//for ( int i = 2; i <= 5; i++ ) {
//	for ( int j = 3; j <= 4; j++ ) {
//		BytePointer biPtr = bi.ptr( i, j );
//		BytePointer b = a.ptr( i - 2, j - 3 );
//		biPtr.fill( b.get() );
//	}
//}
//
//for ( int i = 155; i <= 180; i++ ) {
//	for ( int j = 130; j <= 150; j++ ) {
//		BytePointer ciPtr = bi.ptr( i, j );
//		System.out.println( ciPtr.get() );
//	}
//}
