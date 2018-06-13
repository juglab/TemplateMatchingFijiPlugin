package com.mycompany.imagej;

import java.io.File;
import java.io.IOException;

import org.scijava.command.Command;

import com.indago.io.DoubleTypeImgLoader;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.view.Views;

public class TemplateMatchingPlugin implements Command {

	private static final double pi = 3.14;

	public static void main( String[] args ) throws IOException {
		TemplateMatchingPlugin instanceOfMine = new TemplateMatchingPlugin();
		instanceOfMine.run();
	}

	@Override
	public void run() {
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		String pathName = "/Users/prakash/Desktop/bridge.tif";

//		Dataset imagefile;
//		try {
//			imagefile = ij.scifio().datasetIO().open( pathName );
//		} catch ( IOException e ) {
//			e.printStackTrace();
//			return;
//		}
//		@SuppressWarnings( "unchecked" )
//		ImgPlus< T > img = ( ImgPlus< T > ) imagefile.getImgPlus();

		RandomAccessibleInterval< DoubleType > raiInput = DoubleTypeImgLoader.loadTiffEnsureType( new File( pathName ) );

		long x = -raiInput.dimension( 0 ) / 2;
		long y = -raiInput.dimension( 1 ) / 2;
		AffineTransform2D transform = new AffineTransform2D();
		transform.translate( x, y );
		transform.rotate( pi / 2 );
		transform.translate( -x, -y );


		RealRandomAccessible< DoubleType > realview =
				RealViews.affineReal(
				( Views.interpolate( Views.extendBorder( raiInput ), new NLinearInterpolatorFactory() ) ),
				transform );
		RandomAccessibleInterval< DoubleType > view = Views.interval( Views.raster( realview ), raiInput );
		ij.ui().show( view );
		double[] sigmas = { 1.5, 1.5 };
		//RandomAccessibleInterval< ? extends RealType< ? > > imgSmooth = ij.op().filter().gauss( img, sigmas );
		RandomAccessibleInterval< DoubleType > imgSmooth = ij.op().filter().gauss( raiInput, sigmas );
		ij.ui().show( imgSmooth );


		DoubleType maxVal = ij.op().stats().max( Views.iterable( raiInput ) );
		DoubleType inverse = maxVal.createVariable();
		inverse.setOne();
		inverse.div( maxVal );

		System.out.println( inverse.getRealDouble() );

//		IterableInterval< T > iiSmooth = Views.iterable( imgSmooth );
//		RandomAccess< T > raSmooth = imgSmooth.randomAccess();

//		LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );

		ij.op().math().multiply( Views.iterable( imgSmooth ), inverse );

		ij.ui().show( imgSmooth );
	}

}


//final File imagefile = ij.ui().chooseFile( null, "open" );
//final File templatefile = ij.ui().chooseFile( null, "open" );

// load the dataset
//final Dataset imagedataset = ij.scifio().datasetIO().open( imagefile.getPath() );
//ij.ui().show( imagedataset );

//if ( templatefile != null ) {
//
//	final Dataset templatedataset = ij.scifio().datasetIO().open( templatefile.getPath() );
//	ij.ui().show( templatedataset );
//
//}



//for (int i :imagedataset.size()) {
//	
//}
