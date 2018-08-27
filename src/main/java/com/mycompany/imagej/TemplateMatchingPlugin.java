package com.mycompany.imagej;

import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_imgproc.TM_CCOEFF_NORMED;
import static org.bytedeco.javacpp.opencv_imgproc.matchTemplate;

import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

import ij.IJ;
import ij.ImagePlus;
import ijopencv.ij.ImagePlusMatConverter;
import ijopencv.opencv.MatImagePlusConverter;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
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
import net.imglib2.img.planar.PlanarImgFactory;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Template Matching Plugin for Fiji/ImageJ2
 *
 * @author Mangal Prakash
 */

@Plugin( menuPath = "Plugins>Segmentation>Template Matching Segmentation", type = Command.class )

public class TemplateMatchingPlugin< T extends RealType< T > & NativeType< T > > implements Command {

	@Parameter
	private ImageJ ij;

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
	private OpService ops;

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
		List< RandomAccessibleInterval< T > > trueSegmentationsAndOverlay =
				templateMatching( saveDir, segRadius, imp, template, thresholdmatch, statusService );

		for ( Object results : trueSegmentationsAndOverlay )
			uiService.show( results );
	}

	private < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > templateMatching(
			File saveDir,
			int segRadius,
			RandomAccessibleInterval< T > imp,
			RandomAccessibleInterval< T > template,
			double thresholdmatch,
			StatusService statusService ) {
		final List< RandomAccessibleInterval< T > > imageBucket = sliceImage( imp );

		List< RandomAccessibleInterval< T > > multiTimeSegStack = new ArrayList<>();
		List< RandomAccessibleInterval< T > > multiTimeOverlay = new ArrayList<>();
		int maxStackSize = 0;

		for ( int imageNumber = 0; imageNumber < imageBucket.size(); imageNumber++ ) {
			if ( statusService != null ) {
				statusService.showStatus(
						imageNumber,
						imageBucket.size(),
						"Processing Image" + " " + String.valueOf( imageNumber + 1 ) + "/" + String.valueOf( imageBucket.size() ) );
			}

			RandomAccessibleInterval< T > slice = imageBucket.get( imageNumber );
			Wrapper< T > wrapper =
					calculateNonIntersectSegStackPerTime( segRadius, template, thresholdmatch, slice );
			List< RandomAccessibleInterval< T > > segImagesBucketPerTime = wrapper.listOfSegImages;
			Img< T > overlayPerTime = wrapper.overlayImage;
			maxStackSize = Math.max( maxStackSize, segImagesBucketPerTime.size() );
			RandomAccessibleInterval< T > nonIntersectSegStackPerTime = Views.stack( segImagesBucketPerTime );
			multiTimeSegStack.add( nonIntersectSegStackPerTime );
			multiTimeOverlay.add( overlayPerTime );

		}
		RandomAccessibleInterval< T > overlayStackOverTime = Views.stack( multiTimeOverlay );
//		uiService.show( overlayStackOverTime );

		return createSegmentationOutput( saveDir, imp, multiTimeSegStack, maxStackSize, overlayStackOverTime );
	}


	///All key method implementations below

	private static < T extends RealType< T > & NativeType< T > > List< RandomAccessibleInterval< T > > createSegmentationOutput(
			final File saveDir,
			RandomAccessibleInterval< T > imp,
			List< RandomAccessibleInterval< T > > multiTimeSegStack,
			int maxStackSize,
			RandomAccessibleInterval< T > overlayStackOverTime ) {

		int[] blankImDims = { ( int ) imp.dimension( 0 ), ( int ) imp.dimension( 1 ) };
		RandomAccessibleInterval< T > blankImage = new PlanarImgFactory( Util.getTypeFromInterval( imp ) ).create( blankImDims );
		List< RandomAccessibleInterval< T > > trueSegmentations = new ArrayList<>();
		for ( int index = 0; index < maxStackSize; index++ ) {
			ArrayList< RandomAccessibleInterval< T > > trueSegImageBucket = new ArrayList<>();
			for ( int k = 0; k < multiTimeSegStack.size(); k++ ) {
				RandomAccessibleInterval< T > singleStack = multiTimeSegStack.get( k );
				if ( index >= singleStack.dimension( 2 ) ) {
					trueSegImageBucket.add( blankImage );
				} else {
					RandomAccessibleInterval< T > hyperslice = Views.hyperSlice( singleStack, 2, index );

					trueSegImageBucket.add( hyperslice );
				}
			}
			RandomAccessibleInterval< T > trueSegmentation = Views.stack( trueSegImageBucket );
			trueSegmentations.add( trueSegmentation );
			ImagePlus segPlus = ImageJFunctions.wrap( trueSegmentation, null );
			String strIndex = String.valueOf( index );

			String savePathName = saveDir.getAbsolutePath() + "/" + strIndex + ".tif";

			IJ.save( segPlus, savePathName );
		}
		trueSegmentations.add( overlayStackOverTime );
		return trueSegmentations;
	}

	private < T extends RealType< T > & NativeType< T > > Wrapper< T > calculateNonIntersectSegStackPerTime(
			final int segRadius,
			RandomAccessibleInterval< T > template,
			double thresholdmatch,
			RandomAccessibleInterval< T > raiImg ) {
//		List< Object > detections = new ArrayList<>();
		List< Object > maximaPerTemplate = new ArrayList<>();
		List< Object > anglePerTemplate = new ArrayList<>();

		//Gaussian Smoothing of Image
		T t = Util.getTypeFromInterval( raiImg );
		Img< T > img = ImgView.wrap( raiImg, new ArrayImgFactory<>( t ) );
		Img< T > imgCopy = img.copy();
		RandomAccessibleInterval< T > imgSmooth = gaussSmooth( raiImg );

		//Normalize smoothed image
		normalizeImage( imgSmooth );

		//Converters
		ImagePlusMatConverter ic = new ImagePlusMatConverter();
		MatImagePlusConverter mip = new MatImagePlusConverter();

		ImagePlus wrappedImage = ImageJFunctions.wrap( raiImg, "Original Image" );
		// Convert the image to OpenCV image
		opencv_core.Mat cvImage = ic.convert( wrappedImage, Mat.class );
//		opencv_core.Mat cvSmoothImage = ic.convert( wrappedSmoothImage, Mat.class );

		Img< T > maxHitsIntensity = img.copy();
		LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );
		Img< T > maxHits = maxHitsIntensity.copy();
		Img< T > maxHitsAngle = maxHitsIntensity.copy();
		LoopBuilder.setImages( maxHitsAngle ).forEachPixel( pixel -> pixel.setReal( -1 ) );
//		Img< T > drawImage = maxHitsIntensity.copy(); //Output segmentation image object

		int tH = ( int ) template.dimension( 1 );
		int tW = ( int ) template.dimension( 0 );
		int padHFrom = tH / 2;
		int padWFrom = tW / 2;
		int padHTo = ( int ) ( raiImg.dimension( 1 ) - padHFrom + 1 );
		int padWTo = ( int ) ( raiImg.dimension( 0 ) - padWFrom + 1 );


		for ( int angle = 0; angle < 180; angle = angle + 3 ) {

			//Rotate template
			RandomAccessibleInterval< T > templateRot = Utilities.rotate( ij, template, angle );
			ImagePlus rot = ImageJFunctions.wrap( templateRot, "rotated" );

			// Convert the template to OpenCV image
			opencv_core.Mat cvTemplate = ic.convert( rot, Mat.class );
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

		//Peak Local Maximum detection

		int radius = 1;
		Map< Integer, List > lists = Utilities.peakLocalMax( maxHitsIntensity, radius );
		List< Double > xDetectionsPerTemplatePerTime = lists.get( 1 );
		List< Double > yDetectionsPerTemplatePerTime = lists.get( 2 );

		List< Map.Entry > detectionsPerTemplatePerTime = new ArrayList<>( xDetectionsPerTemplatePerTime.size() );
		if ( yDetectionsPerTemplatePerTime.size() == xDetectionsPerTemplatePerTime.size() ) {

			for ( int i = 0; i < xDetectionsPerTemplatePerTime.size(); i++ ) {
				detectionsPerTemplatePerTime
						.add( new AbstractMap.SimpleEntry( xDetectionsPerTemplatePerTime.get( i ), yDetectionsPerTemplatePerTime.get( i ) ) );
			}
		}

		// probably split here

		System.out.println( detectionsPerTemplatePerTime.size() );
		RandomAccess< T > maxHitsSecondaryAccessor = maxHits.randomAccess();
		RandomAccess< T > maxHitsAngleSecondaryAccessor = maxHitsAngle.randomAccess();

		for ( int i = 0; i < xDetectionsPerTemplatePerTime.size(); i++ ) {
			double xPoint = xDetectionsPerTemplatePerTime.get( i );
			double ypoint = yDetectionsPerTemplatePerTime.get( i );
			maxHitsSecondaryAccessor.setPosition( ( int ) xPoint, 0 );
			maxHitsSecondaryAccessor.setPosition( ( int ) ypoint, 1 );
			maxHitsAngleSecondaryAccessor.setPosition( ( int ) xPoint, 0 );
			maxHitsAngleSecondaryAccessor.setPosition( ( int ) ypoint, 1 );
			maximaPerTemplate.add( maxHitsSecondaryAccessor.get().getRealFloat() );
			anglePerTemplate.add( maxHitsAngleSecondaryAccessor.get().getRealFloat() );

		}

		List< RandomAccessibleInterval< T > > segImagesBucket = new ArrayList< RandomAccessibleInterval< T > >();
		int drawSegRadius = segRadius;
		//Create a list of all zeros to track which coordinates have been plotted
		List< Integer > done = new ArrayList< Integer >( Collections.nCopies( detectionsPerTemplatePerTime.size(), 0 ) );

		boolean repeat = true;
		while ( repeat ) {

			repeat = false;
			Img< T > segImage = img.copy();
			LoopBuilder.setImages( segImage ).forEachPixel( pixel -> pixel.setZero() );

			for ( int i = 0; i < detectionsPerTemplatePerTime.size(); i++ ) {

				if ( done.get( i ) == 1 ) {

					continue;
				}

				int fromRow = ( int ) ( xDetectionsPerTemplatePerTime.get( i ) - drawSegRadius );
				int toRow = ( int ) ( xDetectionsPerTemplatePerTime.get( i ) + drawSegRadius );
				int fromCol = ( int ) ( yDetectionsPerTemplatePerTime.get( i ) - drawSegRadius );
				int toCol = ( int ) ( yDetectionsPerTemplatePerTime.get( i ) + drawSegRadius );

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
					RandomAccess< T > drawingOverlayAccessor = imgCopy.randomAccess();
					double xDrawPoint = xDetectionsPerTemplatePerTime.get( i );
					double yDrawPoint = yDetectionsPerTemplatePerTime.get( i );
					drawingAccessor.setPosition( ( int ) xDrawPoint, 0 );
					drawingAccessor.setPosition( ( int ) yDrawPoint, 1 );
					drawingOverlayAccessor.setPosition( ( int ) xDrawPoint, 0 );
					drawingOverlayAccessor.setPosition( ( int ) yDrawPoint, 1 );
					drawingOverlayAccessor.get().setZero();
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
		Wrapper< T > wrappedOverlayAndSegImgBucket = new Wrapper< T >( segImagesBucket, imgCopy );
		return wrappedOverlayAndSegImgBucket;
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
		//Split the image movie in 2D images and store them in list
		final List< RandomAccessibleInterval< T > > imageBucket =
				new ArrayList< RandomAccessibleInterval< T > >();
		for ( int sliceNumber = 0; sliceNumber < imp.dimension( 2 ); sliceNumber++ ) {
			RandomAccessibleInterval< T > rai = Views.hyperSlice( imp, 2, sliceNumber );
			imageBucket.add( rai );
		}
		return imageBucket;
	}

	public List< RandomAccessibleInterval< T > > calculate(
			RandomAccessibleInterval< T > rawData,
			RandomAccessibleInterval< T > template,
			int segmentationRadius,
			double matchingThreshold ) {
		return templateMatching(
				new File( "/Users/prakash/Desktop/TemplateMatchingSegsButton" ),
				segmentationRadius,
				rawData,
				template,
				matchingThreshold,
				null );
	}

}