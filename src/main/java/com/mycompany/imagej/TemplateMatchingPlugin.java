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
import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class TemplateMatchingPlugin implements Command {

	private static final double pi = 3.14;
	private static final String SetZero = null;

	public static void main( String[] args ) throws IOException {
		TemplateMatchingPlugin instanceOfMine = new TemplateMatchingPlugin();
		instanceOfMine.run();
	}

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
		MatImagePlusConverter mip = new MatImagePlusConverter();

		ImagePlus wrappedImage = ImageJFunctions.wrap( img, "Image" );
		ImagePlus wrappedSmoothImage = ImageJFunctions.wrap( imgSmooth, "Smooth Image" );

		// Convert the image to OpenCV image
		opencv_core.Mat cvImage = ic.convert( wrappedImage, Mat.class );

		ArrayList detectionsPerTemplate = new ArrayList();
		ArrayList maximaPerTemplate = new ArrayList();
		ArrayList anglePerTemplate = new ArrayList();

		ImgPlus< T > maxHitsIntensity = img.copy();
		LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );

		ImgPlus< T > maxHits = maxHitsIntensity.copy();
		ImgPlus< T > maxHitsAngle = maxHitsIntensity.copy();

		for ( int angle = 0; angle <= 60; angle = angle + 30 ) {
			
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

			Img< FloatType > resTimesIntensity = ImageJFunctions.convertFloat( mip.convert( resTimes, ImagePlus.class ) );
			Img< FloatType > ijResults = ImageJFunctions.convertFloat( mip.convert( results, ImagePlus.class ) );
			ArrayList xhits = new ArrayList();
			ArrayList yhits = new ArrayList();

			Cursor< FloatType > cursor = resTimesIntensity.cursor();
			int[] hitCoords = new int[ cursor.numDimensions() ];
			while ( cursor.hasNext() ) {
				cursor.fwd();
				float intensity = cursor.get().getRealFloat();
				if ( intensity >= 0.3 ) {
					cursor.localize( hitCoords );
					xhits.add( hitCoords[ 0 ] );
					yhits.add( hitCoords[ 1 ] );
				}
			}
			
			System.out.println( xhits.size() );
			for ( int i = 0; i < xhits.size(); i++ ) {
//				System.out.println( xhits.get( i ) );
			}

			for ( int i = 0; i < xhits.size(); i++ ) {
				RandomAccess< T > maxHitsIntensityAccessor = maxHitsIntensity.randomAccess();
				RandomAccess< T > resTimesIntensityAccessor = ( RandomAccess< T > ) resTimesIntensity.randomAccess();
				RandomAccess< T > maxHitsAccessor = maxHits.randomAccess();
				RandomAccess< T > maxHitsAngleAccessor = maxHitsAngle.randomAccess();
				RandomAccess< T > resultsAccessor = ( RandomAccess< T > ) ijResults.randomAccess();

				maxHitsIntensityAccessor.setPosition( ( int ) xhits.get( i ), 0 );
				maxHitsIntensityAccessor.setPosition( ( int ) yhits.get( i ), 1 );
				resTimesIntensityAccessor.setPosition( ( int ) xhits.get( i ), 0 );
				resTimesIntensityAccessor.setPosition( ( int ) yhits.get( i ), 1 );
				maxHitsAccessor.setPosition( ( int ) xhits.get( i ), 0 );
				maxHitsAccessor.setPosition( ( int ) yhits.get( i ), 1 );
				resultsAccessor.setPosition( ( int ) xhits.get( i ), 0 );
				resultsAccessor.setPosition( ( int ) yhits.get( i ), 1 );
				maxHitsAngleAccessor.setPosition( ( int ) xhits.get( i ), 0 );
				maxHitsAngleAccessor.setPosition( ( int ) yhits.get( i ), 1 );

				if ( maxHitsIntensityAccessor.get().getRealFloat() < resTimesIntensityAccessor.get().getRealFloat() ) {
					maxHitsIntensityAccessor.get().setReal( resTimesIntensityAccessor.get().getRealFloat() );
					maxHitsAccessor.get().setReal( resultsAccessor.get().getRealFloat() );
					maxHitsAngleAccessor.get().setReal( angle );

				}
			}

//				float a = cursor.get().getRealFloat();
//				if ( a > 0.3 ) {
//					hits.add( cursor.get().getIndex() );
//				}

			

		}


		int radius = 1;
		detectionsPerTemplate = peakLocalMax( maxHitsIntensity, radius, detectionsPerTemplate );
		System.out.println( detectionsPerTemplate.size() );

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


	private < T extends RealType< T > > ArrayList peakLocalMax(
			RandomAccessibleInterval< T > source,
			int radius,
			ArrayList detectionsPerTemplate ) {
		Interval interval = Intervals.expand( source, -1 );
		source = Views.interval( source, interval );
		final Cursor< T > center = Views.iterable( source ).cursor();
		final RectangleShape shape = new RectangleShape( radius, true );
		for ( final Neighborhood< T > localNeighborhood : shape.neighborhoods( source ) ) {
			final T centerValue = center.next();
			boolean isMaximum = true;
			for ( final T value : localNeighborhood ) {
				if ( centerValue.compareTo( value ) <= 0 ) {
					isMaximum = false;
					break;
				}
			}
			if ( isMaximum ) {
				detectionsPerTemplate.add( center );
			}
		}

		return detectionsPerTemplate;

	}

}

