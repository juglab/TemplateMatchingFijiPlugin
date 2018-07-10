package com.mycompany.imagej;

import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgproc.TM_CCOEFF_NORMED;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
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

		List detections = new ArrayList();
		List maxima = new ArrayList();
		List angles = new ArrayList();
		double thresholdmatch = 0.3;

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


		List maximaPerTemplate = new ArrayList();
		List anglePerTemplate = new ArrayList();

		ImgPlus< T > maxHitsIntensity = img.copy();
		LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );

		ImgPlus< T > maxHits = maxHitsIntensity.copy();
		ImgPlus< T > maxHitsAngle = maxHitsIntensity.copy();

		for ( int angle = 0; angle < 30; angle = angle + 30 ) {
			
			//Rotate template
			RandomAccessibleInterval< T > templateRot = rotateAndShow( ij, template, angle );
			ImagePlus rot = ImageJFunctions.wrap( templateRot, "rotated" );

			// Convert the template to OpenCV image
			opencv_core.Mat cvTemplate = ic.convert( rot, Mat.class );
			opencv_core.Mat cvSmoothImage = ic.convert( wrappedSmoothImage, Mat.class );
			opencv_core.Mat temporaryResults = new opencv_core.Mat();

			matchTemplate( cvImage, cvTemplate, temporaryResults, TM_CCOEFF_NORMED );
			normalize( temporaryResults, temporaryResults, 0, 1, NORM_MINMAX, -1, new opencv_core.Mat() );

			//Setting all elements of results matrix to zero
			ImgPlus< T > results = img.copy();
			LoopBuilder.setImages( results ).forEachPixel( pixel -> pixel.setZero() );
			
			Img< FloatType > tempResults = ImageJFunctions.convertFloat( mip.convert(temporaryResults, ImagePlus.class ) );
			RandomAccess< FloatType > subMatrixAccessor = tempResults.randomAccess();
			RandomAccess< T > matrixAccessor = results.randomAccess();
			
			//Replacing the submatrix within results matrix with template matching results matrix

			for (int i =padHFrom; i< padHTo; i++) {
				for (int j = padWFrom; j < padWTo; j++) {
					matrixAccessor.setPosition( i, 0);
					matrixAccessor.setPosition( j, 1 );
					subMatrixAccessor.setPosition( i-padHFrom, 0 );
					subMatrixAccessor.setPosition( j-padWFrom, 1 );
					
					matrixAccessor.get().setReal( subMatrixAccessor.get().getRealDouble() );
				}
					
				}
			
			ImgPlus< T > resCopy = results.copy();

			//Counteracting normalizedness of template matching by multiplying with the smoothed image intensity of raw image
			LoopBuilder.setImages( imgSmooth, results ).forEachPixel( ( a, b ) -> {
				b.mul( a.getRealDouble() );
			} );

			// Finding coordinates where results is greater than a certain threshold(in this case 0.3)
			ArrayList xhits = new ArrayList();
			ArrayList yhits = new ArrayList();

			Cursor< T > cursor = results.cursor();
			int[] hitCoords = new int[ cursor.numDimensions() ];
			while ( cursor.hasNext() ) {
				cursor.fwd();
				double intensity = cursor.get().getRealDouble();
				if ( intensity >= thresholdmatch ) {
					cursor.localize( hitCoords );
					xhits.add( hitCoords[ 0 ] );
					yhits.add( hitCoords[ 1 ] );
				}
			}

//			System.out.println( xhits.size() );
//			for ( int i = 0; i < xhits.size(); i++ ) {
//				System.out.println( xhits.get( i ) );
//			}

			RandomAccess< T > maxHitsIntensityAccessor = maxHitsIntensity.randomAccess();
			RandomAccess< T > resTimesIntensityAccessor = results.randomAccess();
			RandomAccess< T > maxHitsAccessor = maxHits.randomAccess();
			RandomAccess< T > maxHitsAngleAccessor = maxHitsAngle.randomAccess();
			RandomAccess< T > resultsAccessor = resCopy.randomAccess();

			for ( int i = 0; i < xhits.size(); i++ ) {

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

			//Peak Local Maximum detection

			int radius = 1;
			Map< Integer, List > lists = peakLocalMax( maxHitsIntensity, radius );
			List< Double > xDetectionsPerTemplate = lists.get( 1 );
			List< Double > yDetectionsPerTemplate = lists.get( 2 );

			List< Map.Entry > detectionsPerTemplate = new ArrayList<>( xDetectionsPerTemplate.size() );
			if ( yDetectionsPerTemplate.size() == xDetectionsPerTemplate.size() ) {

				for ( int i = 0; i < xDetectionsPerTemplate.size(); i++ ) {
					detectionsPerTemplate.add( new AbstractMap.SimpleEntry( xDetectionsPerTemplate.get( i ), yDetectionsPerTemplate.get( i ) ) );
				}
			}

			System.out.println( xDetectionsPerTemplate.size() );

//			for ( int i = 0; i < 10; i++ ) {
//				Entry< Double, Double > entry = detectionsPerTemplate.get( i );
//				System.out.println( entry );
//			}

			RandomAccess< T > maxHitsSecondaryAccessor = maxHits.randomAccess();
			RandomAccess< T > maxHitsAngleSecondaryAccessor = maxHitsAngle.randomAccess();

			for ( int i = 0; i < xDetectionsPerTemplate.size(); i++ ) {
				double xPoint = xDetectionsPerTemplate.get( i );
				double ypoint = yDetectionsPerTemplate.get( i );
				maxHitsSecondaryAccessor.setPosition( ( int ) xPoint, 0 );
				maxHitsSecondaryAccessor.setPosition( ( int ) ypoint, 1 );
				maxHitsAngleSecondaryAccessor.setPosition( ( int ) xPoint, 0 );
				maxHitsAngleSecondaryAccessor.setPosition( ( int ) ypoint, 1 );
				maximaPerTemplate.add( maxHitsSecondaryAccessor.get().getRealFloat() );
				anglePerTemplate.add( maxHitsAngleSecondaryAccessor.get().getRealFloat() );

			}

			detections.addAll( detectionsPerTemplate );
			maxima.addAll( maximaPerTemplate );
			angles.addAll( anglePerTemplate );

			for ( int j = 0; j < 10; j++ ) {
				System.out.println( maxima.get( j ) );
			}
		}

			//Test code begins

//			RandomAccess< T > testAccessor = results.randomAccess();
//			for ( int i = 0; i < results.dimension( 0 ); i++ ) {
//				for ( int j = 0; j < results.dimension( 1 ); j++ ) {
//					testAccessor.setPosition( i, 0 );
//					testAccessor.setPosition( j, 1 );
//					System.out.println( testAccessor.get().getRealDouble() );
//				}
//
//			}
//
//		}
		

			//Test code ends

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


	private < T extends RealType< T > > Map< Integer, List > peakLocalMax(
			RandomAccessibleInterval< T > source,
			int radius ) {

		Map< Integer, List > listMap = new HashMap< Integer, List >();
		List xArray1 = new ArrayList();
		List yArray1 = new ArrayList();
		listMap.put( 1, xArray1 );
		listMap.put( 2, yArray1 );

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

				xArray1.add( ( double ) center.getIntPosition( 0 ) );
				yArray1.add( ( double ) center.getIntPosition( 1 ) );
			}
		}

		return listMap;

	}

}

