package com.mycompany.imagej;

import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgproc.TM_CCOEFF_NORMED;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
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
import net.imglib2.img.ImgView;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

public class TemplateMatchingPlugin implements Command {

	private static final double pi = 3.14;

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


	private < T extends RealType< T > & NativeType< T > > void runThrowsException() throws Exception {
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		String imagePathName = "/Users/prakash/Desktop/BeetlesDataAndResults/Tr2D10time/raw.tif";
		Dataset imagefile = ij.scifio().datasetIO().open( imagePathName );
		ImgPlus< T > imp = ( ImgPlus< T > ) imagefile.getImgPlus();

		//Split the image movie in 2D images and store them in list
		final List< RandomAccessibleInterval< T > > imageBucket =
				new ArrayList< RandomAccessibleInterval< T > >();
		for ( int sliceNumber = 0; sliceNumber < imp.dimension( 2 ); sliceNumber++ ) {
			RandomAccessibleInterval< T > rai = Views.hyperSlice( imp, 2, sliceNumber );
			imageBucket.add( rai );
		}

		//List to store templates
		final List< RandomAccessibleInterval< T > > templateBucket =
				new ArrayList< RandomAccessibleInterval< T > >();

		//Load template(s) 
		String templatePathName = "/Users/prakash/Desktop/raw_untemp1.tif";
		Dataset templatefile = ij.scifio().datasetIO().open( templatePathName );
		ImgPlus< T > templateFirst = ( ImgPlus< T > ) templatefile.getImgPlus();
		templateBucket.add( templateFirst );

		String templatePathNameSecond = "/Users/prakash/Desktop/raw_untemp2.tif";
		Dataset templatefileSecond = ij.scifio().datasetIO().open( templatePathNameSecond );
		ImgPlus< T > templateSecond = ( ImgPlus< T > ) templatefileSecond.getImgPlus();
		templateBucket.add( templateSecond );

		double thresholdmatch = 0.3;

		for ( int imageNumber = 0; imageNumber < imageBucket.size(); imageNumber++ ) {

			List detections = new ArrayList();
			List xDetections = new ArrayList();
			List yDetections = new ArrayList();
			List maxima = new ArrayList();
			List angles = new ArrayList();

			//Gaussian Smoothing of Image
			RandomAccessibleInterval< T > raiImg = imageBucket.get( imageNumber );
			T t = Util.getTypeFromInterval( raiImg );
			Img< T > img = ImgView.wrap( raiImg, new ArrayImgFactory<>( t ) );
			Img< T > imgCopy = img.copy();
//			ImgPlus< T > imgPlus = new ImgPlus<>( img );
//			ArrayImgFactory< T > factory = new ArrayImgFactory<>( t );
//			factory.create( 10, 10 );
			//			ImgPlus img = ( ImgPlus ) ImageJFunctions.wrapFloat( ImageJFunctions.show( raiImg ) );


			double[] sigmas = { 1.5, 1.5 };
			RandomAccessibleInterval< FloatType > imgSmooth = ij.op().filter().gauss( raiImg, sigmas );
			System.out.println( Util.getTypeFromInterval( imgSmooth ).getClass() );

			FloatType maxVal = new FloatType();
			ij.op().stats().max( maxVal, Views.iterable( imgSmooth ) );
			float inverse = 1.0f / maxVal.getRealFloat();

			//Normalizing the image
			LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );
			ij.ui().show( imgSmooth );

			//Converters
			ImagePlusMatConverter ic = new ImagePlusMatConverter();
			MatImagePlusConverter mip = new MatImagePlusConverter();

			ImagePlus wrappedImage = ImageJFunctions.wrap( raiImg, "Original Image" );
			ImagePlus wrappedSmoothImage = ImageJFunctions.wrap( imgSmooth, "Smooth Image" );
			// Convert the image to OpenCV image
			opencv_core.Mat cvImage = ic.convert( wrappedImage, Mat.class );
			opencv_core.Mat cvSmoothImage = ic.convert( wrappedSmoothImage, Mat.class );


			for ( int templateNumber = 0; templateNumber < templateBucket.size(); templateNumber++ ) {

				List maximaPerTemplate = new ArrayList();
				List anglePerTemplate = new ArrayList();

				Img< T > maxHitsIntensity = img.copy();
				LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );
				Img< T > maxHits = maxHitsIntensity.copy();
				Img< T > maxHitsAngle = maxHitsIntensity.copy();
				LoopBuilder.setImages( maxHitsAngle ).forEachPixel( pixel -> pixel.setReal( -1 ) );
				Img< T > drawImage = maxHitsIntensity.copy(); //Output segmentation image

				RandomAccessibleInterval< T > template = templateBucket.get( templateNumber );
				int tH = ( int ) template.dimension( 1 );
				int tW = ( int ) template.dimension( 0 );
				int padHFrom = tH / 2;
				int padWFrom = tW / 2;
				int padHTo = ( int ) ( raiImg.dimension( 1 ) - padHFrom + 1 );
				int padWTo = ( int ) ( raiImg.dimension( 0 ) - padWFrom + 1 );

				for ( int angle = 0; angle < 180; angle = angle + 3 ) {
					//Rotate template
					RandomAccessibleInterval< T > templateRot = rotate( ij, template, angle );
					ImagePlus rot = ImageJFunctions.wrap( templateRot, "rotated" );

					// Convert the template to OpenCV image
					opencv_core.Mat cvTemplate = ic.convert( rot, Mat.class );
					opencv_core.Mat temporaryResults = new opencv_core.Mat();

					matchTemplate( cvImage, cvTemplate, temporaryResults, TM_CCOEFF_NORMED );
					normalize( temporaryResults, temporaryResults, 0, 1, NORM_MINMAX, -1, new opencv_core.Mat() );

//					//Setting all elements of results matrix to zero
					Img< T > results = img.copy();
					LoopBuilder.setImages( results ).forEachPixel( pixel -> pixel.setZero() );

					Img< FloatType > tempResults = ImageJFunctions.convertFloat( mip.convert( temporaryResults, ImagePlus.class ) );
					RandomAccess< FloatType > subMatrixAccessor = tempResults.randomAccess();
					RandomAccess< T > matrixAccessor = results.randomAccess();

					//Replacing the submatrix within results matrix with template matching results matrix

					for ( int i = padHFrom; i < padHTo; i++ ) {
						for ( int j = padWFrom; j < padWTo; j++ ) {
							matrixAccessor.setPosition( i, 0 );
							matrixAccessor.setPosition( j, 1 );
							subMatrixAccessor.setPosition( i - padHFrom, 0 );
							subMatrixAccessor.setPosition( j - padWFrom, 1 );

							matrixAccessor.get().setReal( subMatrixAccessor.get().getRealDouble() );
						}

					}

					Img< T > resCopy = results.copy();

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
				}
				//Peak Local Maximum detection

				int radius = 1;

				Map< Integer, List > lists = peakLocalMax( maxHitsIntensity, radius );
				List< Double > xDetectionsPerTemplate = lists.get( 1 );
				List< Double > yDetectionsPerTemplate = lists.get( 2 );

				List< Map.Entry > detectionsPerTemplate = new ArrayList<>( xDetectionsPerTemplate.size() );
				if ( yDetectionsPerTemplate.size() == xDetectionsPerTemplate.size() ) {

					for ( int i = 0; i < xDetectionsPerTemplate.size(); i++ ) {
						detectionsPerTemplate
								.add( new AbstractMap.SimpleEntry( xDetectionsPerTemplate.get( i ), yDetectionsPerTemplate.get( i ) ) );
					}
				}

//				System.out.println( xDetectionsPerTemplate.size() );
				System.out.println( detectionsPerTemplate.size() );

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
//
				detections.addAll( detectionsPerTemplate );
				xDetections.addAll( xDetectionsPerTemplate );
				yDetections.addAll( yDetectionsPerTemplate );
				maxima.addAll( maximaPerTemplate );
				angles.addAll( anglePerTemplate );
				System.out.println( detections.size() );
//
//				/// Draw segmentations
//
//				RandomAccess< T > drawingAccessor = drawImage.randomAccess();
//				for ( int i = 0; i < detections.size(); i++ ) {
//					double xDrawPoint = xDetectionsPerTemplate.get( i );
//					double yDrawPoint = yDetectionsPerTemplate.get( i );
//					drawingAccessor.setPosition( ( int ) xDrawPoint, 0 );
//					drawingAccessor.setPosition( ( int ) yDrawPoint, 1 );
//
//					HyperSphere< T > hyperSphere = new HyperSphere<>( drawImage, drawingAccessor, 1 );
//						// set every value inside the sphere to 1
//					for ( T value : hyperSphere )
//							value.setOne();
//				}
//
//				ij.ui().show( drawImage );

			}
			System.out.println( "done!" );

			List segImagesBucket = new ArrayList();
			int drawSegRadius = 4;
			Img< T > intersectingSegs = img.copy();
			LoopBuilder.setImages( intersectingSegs).forEachPixel( pixel -> pixel.setZero() );
			
			//Create a list of all zeros to track which coordinates have been plotted
			List< Integer > done = new ArrayList< Integer >( Collections.nCopies( detections.size(), 0 ) );

			for ( int i = 0; i < detections.size(); i++ ) {
				if ( done.get( i ) == 1 ) {
					continue;
				}
				
				int fromRow = ( int ) xDetections.get( i ) - drawSegRadius;
				int toRow = ( int ) xDetections.get( i ) + drawSegRadius;
				int fromCol = ( int ) yDetections.get( i ) - drawSegRadius;
				int toCol = ( int ) yDetections.get( i ) + drawSegRadius;
				
				double searchMax = 0;
				RandomAccess< T > intersectingSegsAccessor = intersectingSegs.randomAccess();

				for ( int row = fromRow; row < toRow; row++ ) {
					for ( int col = fromCol; col < toCol; col++ ) {
						intersectingSegsAccessor.setPosition( row, 0 );
						intersectingSegsAccessor.setPosition( col, 1 );
						if ( intersectingSegsAccessor.get().getRealDouble() > 0 ) {
							searchMax = intersectingSegsAccessor.get().getRealDouble();
						}
					}
				}
				if ( searchMax == 0 ) {
					//Draw segmentation circle of radius drawSegRadius on intersectingSegs and set
					//done.get(i) ==1
				}
			}

		}

	}


	private < T extends RealType< T > > Map< Integer, List > peakLocalMax(
			RandomAccessibleInterval< T > source,
			int radius ) {

		Map< Integer, List > listMap = new HashMap< Integer, List >();
		List xArray1 = new ArrayList();
		List yArray1 = new ArrayList();
		List centerList = new ArrayList();
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
				centerList.add( centerValue );
			}
		}

		return listMap;
	}


	private < T extends RealType< T > > RandomAccessibleInterval< T > rotate(
			final ImageJ ij,
			RandomAccessibleInterval< T > template,
			int angle ) {
		long x = -template.dimension( 0 ) / 2;
		long y = -template.dimension( 1 ) / 2;
		AffineTransform2D transform = new AffineTransform2D();
		transform.translate( x, y );
		transform.rotate( angle );
		transform.translate( -x, -y );
		RealRandomAccessible< T > realview =
				RealViews.affineReal(
						( Views.interpolate( Views.extendBorder( template ), new NLinearInterpolatorFactory() ) ),
						transform );
		RandomAccessibleInterval< T > view = Views.interval( Views.raster( realview ), template );
//		ij.ui().show( view );
		return ( view );
	}


}
