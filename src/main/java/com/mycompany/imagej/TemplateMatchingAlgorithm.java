package com.mycompany.imagej;

import ij.IJ;
import ij.ImagePlus;
import ijopencv.ij.ImagePlusMatConverter;
import ijopencv.opencv.MatImagePlusConverter;
import net.imagej.ops.OpService;
import net.imglib2.Cursor;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.region.hypersphere.HyperSphere;
import net.imglib2.img.Img;
import net.imglib2.img.ImgView;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;
import org.bytedeco.javacpp.opencv_core;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.plugin.Parameter;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgproc.TM_CCOEFF_NORMED;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;

public class TemplateMatchingAlgorithm
{
	@Parameter
	private OpService ops;

	public TemplateMatchingAlgorithm( Context context ) {
		context.inject( this );
	}

	public < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > calculate(
			RandomAccessibleInterval< T > imp,
			RandomAccessibleInterval< T > template,
			int segRadius,
			double thresholdmatch,
			StatusService statusService ) {
		final List< RandomAccessibleInterval< T > > imageBucket = sliceImage( imp );

		List< RandomAccessibleInterval< T > > multiTimeSegStack = new ArrayList<>();

		for ( int imageNumber = 0; imageNumber < imageBucket.size(); imageNumber++ ) {
			if ( statusService != null ) {
				statusService.showStatus(
						imageNumber,
						imageBucket.size(),
						"Processing Image" + " " + ( imageNumber + 1 ) + "/" + ( imageBucket.size() ) );
			}

			RandomAccessibleInterval< T > slice = imageBucket.get( imageNumber );
			multiTimeSegStack.add( calculateFrame( segRadius, template, thresholdmatch, slice ) );
		}

		return reslice( multiTimeSegStack );
	}

	private < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval< T > calculateFrame( int segRadius, RandomAccessibleInterval< T > template, double thresholdmatch, RandomAccessibleInterval< T > slice )
	{
		ArrayList< Point > points =
				detectionCoordsPerTime( template, thresholdmatch, slice );
		List< RandomAccessibleInterval< T > > segImagesBucketPerTime =
				drawNonIntersectingPoints( slice, segRadius, points );
		return Views.stack( segImagesBucketPerTime );
	}

	///All key method implementations below

	private static < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > reslice(
			List< RandomAccessibleInterval< T > > frames ) {
		long maxStackSize = 0;
		for(RandomAccessibleInterval< T > frame : frames)
			maxStackSize = Math.max( maxStackSize, frame.dimension(2) );
		List< RandomAccessibleInterval< T > > channels = new ArrayList<>();
		for ( int index = 0; index < maxStackSize; index++ ) {
			channels.add( extractChannel( frames, index ) );
		}
		return channels;
	}

	private static < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval< T > extractChannel( List< RandomAccessibleInterval< T > > frames, int index )
	{
		RandomAccessibleInterval< T > blankImage = new ArrayImgFactory<>( Util.getTypeFromInterval( frames.get( 0 ) ) )
				.create( frames.get(0).dimension( 0 ), frames.get(0).dimension( 1 ) );
		ArrayList< RandomAccessibleInterval< T > > trueSegImageBucket = new ArrayList<>();
		for ( int k = 0; k < frames.size(); k++ ) {
			RandomAccessibleInterval< T > frame = frames.get( k );
			RandomAccessibleInterval< T > a = ( index >= frame.dimension( 2 ) ) ?
					blankImage : Views.hyperSlice( frame, 2, index );
			trueSegImageBucket.add( a );
		}
		return Views.stack( trueSegImageBucket );
	}

	private < T extends RealType< T > & NativeType< T > > ArrayList< Point > detectionCoordsPerTime(
			RandomAccessibleInterval< T > template,
			double thresholdmatch,
			RandomAccessibleInterval< T > raiImg ) {
		Map< Integer, Img< T > > hitmap = computeThresholdedMatchingPoints( template, thresholdmatch, raiImg ); //Return maxHitsIntensity and maxHits
		Img< T > maxHitsIntensity = hitmap.get( 1 );

		//Peak Local Maximum detection
		int radius = 1;
		ArrayList< Point > localMaximaCoords = Utilities.peakLocalMax( maxHitsIntensity, radius );
		System.out.println( localMaximaCoords.size() );
		return localMaximaCoords;
	}

	private < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > drawNonIntersectingPoints(
			RandomAccessibleInterval< T > raiImg,
			final int segRadius,
			ArrayList< Point > localMaximaCoords ) {

		List< RandomAccessibleInterval< T > > segImagesBucket = new ArrayList< RandomAccessibleInterval< T > >();

		int drawSegRadius = segRadius;
		//Create a list of all zeros to track which coordinates have been plotted
		List< Integer > done = new ArrayList< Integer >( Collections.nCopies( localMaximaCoords.size(), 0 ) );

		boolean repeat = true;
		while ( repeat ) {

			repeat = false;
			RandomAccessibleInterval< T > segImage = ops.copy().rai( raiImg );

			LoopBuilder.setImages( segImage ).forEachPixel( pixel -> pixel.setZero() );


			for ( int i = 0; i < localMaximaCoords.size(); i++ ) {

				if ( done.get( i ) == 1 ) {

					continue;
				}
				int fromRow = ( localMaximaCoords.get( i ).getIntPosition( 0 ) - drawSegRadius );
				int toRow = ( localMaximaCoords.get( i ).getIntPosition( 0 ) + drawSegRadius );
				int fromCol = ( localMaximaCoords.get( i ).getIntPosition( 1 ) - drawSegRadius );
				int toCol = ( localMaximaCoords.get( i ).getIntPosition( 1 ) + drawSegRadius );

				double searchMax = 0;
				RandomAccess< T > intersectingSegsAccessor = segImage.randomAccess();

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

					RandomAccess< T > drawingAccessor = segImage.randomAccess();
					///////////Check here too!
					int xDrawPoint = localMaximaCoords.get( i ).getIntPosition( 0 );
					int yDrawPoint = localMaximaCoords.get( i ).getIntPosition( 1 );
					//////////
					drawingAccessor.setPosition( xDrawPoint, 0 );
					drawingAccessor.setPosition( yDrawPoint, 1 );
					HyperSphere< T > hyperSphere = new HyperSphere<>( segImage, drawingAccessor, drawSegRadius );

					// set every value inside the sphere to 1
					for ( T value : hyperSphere ) {
						value.setOne();
					}
					done.set( i, 1 );
					repeat = true;

				}

			}
			if ( repeat ) {
				segImagesBucket.add( segImage );
			}
		}
		return segImagesBucket;
	}

	private < T extends RealType< T > & NativeType< T > > Map< Integer, Img< T > > computeThresholdedMatchingPoints(
			RandomAccessibleInterval< T > template,
			double thresholdmatch,
			RandomAccessibleInterval< T > raiImg ) {

		T t = Util.getTypeFromInterval( raiImg );
		Img< T > img = ImgView.wrap( raiImg, new ArrayImgFactory<>( t ) );
		//Gaussian Smoothing of Image
		RandomAccessibleInterval< T > imgSmooth = gaussSmooth( raiImg );

		//Normalize smoothed image
		normalizeImage( imgSmooth );

		ImagePlus wrappedImage = ImageJFunctions.wrap( raiImg, "Original Image" );

		//Converters
		ImagePlusMatConverter ic = new ImagePlusMatConverter();
		MatImagePlusConverter mip = new MatImagePlusConverter();
		// Convert the image to OpenCV image
		opencv_core.Mat cvImage = ic.convert( wrappedImage, opencv_core.Mat.class );

		Img< T > maxHitsIntensity = img.copy();
		LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );
		Img< T > maxHits = maxHitsIntensity.copy();
		Img< T > maxHitsAngle = maxHitsIntensity.copy();
		LoopBuilder.setImages( maxHitsAngle ).forEachPixel( pixel -> pixel.setReal( -1 ) );

		int tH = ( int ) template.dimension( 1 );
		int tW = ( int ) template.dimension( 0 );
		int padHFrom = tH / 2;
		int padWFrom = tW / 2;
		int padHTo = ( int ) ( raiImg.dimension( 1 ) - padHFrom + 1 );
		int padWTo = ( int ) ( raiImg.dimension( 0 ) - padWFrom + 1 );


		for ( int angle = 0; angle < 180; angle = angle + 3 ) {

			//Rotate template
			RandomAccessibleInterval< T > templateRot = Utilities.rotate( template, angle );
			ImagePlus rot = ImageJFunctions.wrap( templateRot, "rotated" );

			// Convert the template to OpenCV image
			opencv_core.Mat cvTemplate = ic.convert( rot, opencv_core.Mat.class );
			opencv_core.Mat temporaryResults = new opencv_core.Mat();

			matchTemplate( cvImage, cvTemplate, temporaryResults, TM_CCOEFF_NORMED );
			normalize( temporaryResults, temporaryResults, 0, 1, NORM_MINMAX, -1, new opencv_core.Mat() );

			//Setting all elements of results matrix to zero
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
			ArrayList< Point > hits = new ArrayList<>();

			Cursor< T > cursor = results.cursor();
			while ( cursor.hasNext() ) {
				cursor.fwd();
				double intensity = cursor.get().getRealDouble();
				if ( intensity >= thresholdmatch ) {
					Point hitCoords = new Point( cursor.numDimensions() );
					hitCoords.setPosition( cursor );
					hits.add( hitCoords );
				}
			}

			RandomAccess< T > maxHitsIntensityAccessor = maxHitsIntensity.randomAccess();
			RandomAccess< T > resTimesIntensityAccessor = results.randomAccess();
			RandomAccess< T > maxHitsAccessor = maxHits.randomAccess();
			RandomAccess< T > maxHitsAngleAccessor = maxHitsAngle.randomAccess();
			RandomAccess< T > resultsAccessor = resCopy.randomAccess();

			for ( int i = 0; i < hits.size(); i++ ) {

				maxHitsIntensityAccessor.setPosition( hits.get( i ) );
				resTimesIntensityAccessor.setPosition( hits.get( i ) );
				maxHitsAccessor.setPosition( hits.get( i ) );
				resultsAccessor.setPosition( hits.get( i ) );
				maxHitsAngleAccessor.setPosition( hits.get( i ) );

				if ( maxHitsIntensityAccessor.get().getRealFloat() < resTimesIntensityAccessor.get().getRealFloat() ) {
					maxHitsIntensityAccessor.get().setReal( resTimesIntensityAccessor.get().getRealFloat() );
					maxHitsAccessor.get().setReal( resultsAccessor.get().getRealFloat() );
					maxHitsAngleAccessor.get().setReal( angle );

				}
			}
		}
		Map< Integer, Img< T > > hitMap = new HashMap< Integer, Img< T > >();
		hitMap.put( 1, maxHitsIntensity );
		hitMap.put( 2, maxHits );
		return hitMap;

	}


	private < T extends RealType< T > > void normalizeImage( RandomAccessibleInterval< T > imgSmooth ) {
		T maxVal = Util.getTypeFromInterval( imgSmooth ).createVariable();
		ops.stats().max( maxVal, Views.iterable( imgSmooth ) );
		float inverse = 1.0f / maxVal.getRealFloat();
		//Normalizing the image
		LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );
	}

	private < T extends RealType< T > & NativeType< T > > RandomAccessibleInterval< T > gaussSmooth( RandomAccessibleInterval< T > raiImg ) {
		double[] sigmas = { 1.5, 1.5 };
		return ops.filter().gauss( raiImg, sigmas );
	}

	private static < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > sliceImage(
			RandomAccessibleInterval< T > imp ) {
		if ( imp.numDimensions() > 2 ) {
			//Split the image movie in 2D images and store them in list
			final List< RandomAccessibleInterval< T > > imageBucket = new ArrayList<>();
			for ( int sliceNumber = 0; sliceNumber < imp.dimension( 2 ); sliceNumber++ ) {
				RandomAccessibleInterval< T > rai = Views.hyperSlice( imp, 2, sliceNumber );
				imageBucket.add( rai );
			}
			return imageBucket;
		} else {
			return Collections.singletonList( imp );
		}

	}
}
