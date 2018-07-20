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

import ij.IJ;
import ij.ImagePlus;
import ijopencv.ij.ImagePlusMatConverter;
import ijopencv.opencv.MatImagePlusConverter;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.Cursor;
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

/**
 * Template Matching Plugin for Fiji/ImageJ2
 *
 * @author Mangal Prakash
 */

@Plugin( menuPath = "Plugins>Segmentation>Template Matching Segmentation", type = Command.class )

public class TemplateMatchingPlugin implements Command {

	@Parameter
	private ImageJ ij;

	@Parameter
	DatasetIOService datasetIOService;

//	@Parameter
//	private Dataset dataset;
	
	@Parameter( label = "Image to load" )
	private File inputImage;

	@Parameter( label = "Template to load" )
	private File inputTemplate;

	@Parameter( style = "directory" )
	private File saveResultsDir;

	@Parameter( label = "Segmentation circle radius" )
	private int segCircleRad;

	@Parameter
	StatusService statusService;

	public static void main( String[] args ) throws IOException {

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
//		final File inputFile = ij.ui().chooseFile( null, "open" );
//		final Dataset dataset = ij.scifio().datasetIO().open( inputFile.getPath() );
//		ij.ui().show( dataset );
		ij.command().run( TemplateMatchingPlugin.class, true );

	}

	@Override
	public void run() {
		try {
			Dataset imageFile = datasetIOService.open( inputImage.getAbsolutePath() );
			Dataset templateFile = datasetIOService.open( inputTemplate.getAbsolutePath() );
			File saveDir = saveResultsDir;
			int segRad = segCircleRad;
			
//			if(saveDir.isDirectory()) {
//				if(saveDir.list().length  == 0) {
//					saveDir = saveResultsDir;
//				}
//				else {
//					System.out.println( "Directory not empty! Please choose another directory" );
//					saveDir = ij.ui().chooseFile( null, "save" );
//				}
//			}
			
			templateMatching( imageFile, templateFile, saveDir, segRad );
		} catch ( Exception e ) {
			e.printStackTrace();
		}

	}


	private < T extends RealType< T > & NativeType< T > > void templateMatching(
			final Dataset imagefile,
			final Dataset templateFile,
			final File saveDir,
			final int segRadius )
			throws Exception {
		
		ImgPlus< T > imp = ( ImgPlus< T > ) imagefile.getImgPlus();
		
		//Split the image movie in 2D images and store them in list
		final List< RandomAccessibleInterval< T > > imageBucket =
				new ArrayList< RandomAccessibleInterval< T > >();
		for ( int sliceNumber = 0; sliceNumber < imp.dimension( 2 ); sliceNumber++ ) {
			RandomAccessibleInterval< T > rai = Views.hyperSlice( imp, 2, sliceNumber );
			imageBucket.add( rai );
		}

		ImgPlus< T > template = ( ImgPlus< T > ) templateFile.getImgPlus();


		double thresholdmatch = 0.3;
		List multiTimeStack = new ArrayList();
		int maxStackSize = 0;

		for ( int imageNumber = 0; imageNumber < imageBucket.size(); imageNumber++ ) {

			statusService.showStatus(
					imageNumber,
					imageBucket.size(),
					"Processing Image" + " " + String.valueOf( imageNumber ) + "/" + String.valueOf( imageBucket.size() ) );
			List detections = new ArrayList();
			List xDetections = new ArrayList();
			List yDetections = new ArrayList();
			List maximaPerTemplate = new ArrayList();
			List anglePerTemplate = new ArrayList();


			//Gaussian Smoothing of Image
			RandomAccessibleInterval< T > raiImg = imageBucket.get( imageNumber );
			T t = Util.getTypeFromInterval( raiImg );
			Img< T > img = ImgView.wrap( raiImg, new ArrayImgFactory<>( t ) );
			Img< T > imgCopy = img.copy();

			double[] sigmas = { 1.5, 1.5 };
			RandomAccessibleInterval< FloatType > imgSmooth = ij.op().filter().gauss( raiImg, sigmas );
			FloatType maxVal = new FloatType();
			ij.op().stats().max( maxVal, Views.iterable( imgSmooth ) );
			float inverse = 1.0f / maxVal.getRealFloat();

			//Normalizing the image
			LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );

			//Converters
			ImagePlusMatConverter ic = new ImagePlusMatConverter();
			MatImagePlusConverter mip = new MatImagePlusConverter();

			ImagePlus wrappedImage = ImageJFunctions.wrap( raiImg, "Original Image" );
			ImagePlus wrappedSmoothImage = ImageJFunctions.wrap( imgSmooth, "Smooth Image" );

			// Convert the image to OpenCV image
			opencv_core.Mat cvImage = ic.convert( wrappedImage, Mat.class );
			opencv_core.Mat cvSmoothImage = ic.convert( wrappedSmoothImage, Mat.class );

			Img< T > maxHitsIntensity = img.copy();
			LoopBuilder.setImages( maxHitsIntensity ).forEachPixel( pixel -> pixel.setZero() );
			Img< T > maxHits = maxHitsIntensity.copy();
			Img< T > maxHitsAngle = maxHitsIntensity.copy();
			LoopBuilder.setImages( maxHitsAngle ).forEachPixel( pixel -> pixel.setReal( -1 ) );
			Img< T > drawImage = maxHitsIntensity.copy(); //Output segmentation image

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
			Map< Integer, List > lists = Utilities.peakLocalMax( maxHitsIntensity, radius );
			List< Double > xDetectionsPerTemplate = lists.get( 1 );
			List< Double > yDetectionsPerTemplate = lists.get( 2 );

			List< Map.Entry > detectionsPerTemplate = new ArrayList<>( xDetectionsPerTemplate.size() );
			if ( yDetectionsPerTemplate.size() == xDetectionsPerTemplate.size() ) {

				for ( int i = 0; i < xDetectionsPerTemplate.size(); i++ ) {
					detectionsPerTemplate
							.add( new AbstractMap.SimpleEntry( xDetectionsPerTemplate.get( i ), yDetectionsPerTemplate.get( i ) ) );
				}
			}

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
			xDetections.addAll( xDetectionsPerTemplate );
			yDetections.addAll( yDetectionsPerTemplate );

			System.out.println( "done!" );

			List segImagesBucket = new ArrayList();
			int drawSegRadius = segRadius;
			//Create a list of all zeros to track which coordinates have been plotted
			List< Integer > done = new ArrayList< Integer >( Collections.nCopies( detections.size(), 0 ) );

			boolean repeat = true;
			while ( repeat ) {

				repeat = false;
				Img< T > segImage = img.copy();
				LoopBuilder.setImages( segImage ).forEachPixel( pixel -> pixel.setZero() );

				for ( int i = 0; i < detections.size(); i++ ) {

					if ( done.get( i ) == 1 ) {

						continue;
					}

					int fromRow = ( int ) ( xDetectionsPerTemplate.get( i ) - drawSegRadius );
					int toRow = ( int ) ( xDetectionsPerTemplate.get( i ) + drawSegRadius );
					int fromCol = ( int ) ( yDetectionsPerTemplate.get( i ) - drawSegRadius );
					int toCol = ( int ) ( yDetectionsPerTemplate.get( i ) + drawSegRadius );

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
						double xDrawPoint = xDetectionsPerTemplate.get( i );
						double yDrawPoint = yDetectionsPerTemplate.get( i );
						drawingAccessor.setPosition( ( int ) xDrawPoint, 0 );
						drawingAccessor.setPosition( ( int ) yDrawPoint, 1 );
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
					int stackSize = segImagesBucket.size();
					if(stackSize > maxStackSize ) {
						maxStackSize = stackSize;
					}

				}

			}
			RandomAccessibleInterval oneTimeStack = Views.stack( segImagesBucket );
			multiTimeStack.add( oneTimeStack );

		}
		System.out.println( maxStackSize );
		System.out.println( multiTimeStack.size() );
		RandomAccessibleInterval trueSegmentation = null;

		int[] blankImDims = { ( int ) imp.dimension( 0 ), ( int ) imp.dimension( 1 ) };
		RandomAccessibleInterval< T > blankImage = imp.getImg().factory().create( blankImDims );

		for ( int index = 0; index < maxStackSize; index++ ) {
			ArrayList trueSegImageBucket = new ArrayList();
			for ( int k = 0; k < multiTimeStack.size(); k++ ) {
				RandomAccessibleInterval singleStack = ( RandomAccessibleInterval ) multiTimeStack.get( k );
				if ( index >= singleStack.dimension( 2 ) ) {
					trueSegImageBucket.add( blankImage );
				}
				else {
					RandomAccessibleInterval hyperslice = Views.hyperSlice( singleStack, 2, index );

					trueSegImageBucket.add( hyperslice );
				}
				trueSegmentation = Views.stack( trueSegImageBucket );
			}
			ij.ui().show( trueSegmentation );
			ImagePlus segPlus = ImageJFunctions.wrap( trueSegmentation, null );
			String strIndex = String.valueOf( index );

			String savePathName = saveDir.getAbsolutePath() + "/" + strIndex + ".tif";

			IJ.save( segPlus, savePathName );

//			ij.scifio().datasetIO().save( ds.create( trueSegmentation ), savePathName );

		}
	}
}
