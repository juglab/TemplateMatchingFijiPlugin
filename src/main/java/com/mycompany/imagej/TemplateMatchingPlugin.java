package com.mycompany.imagej;

import java.io.IOException;

import org.scijava.command.Command;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
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
		String pathName = "/Users/prakash/Desktop/bridge32.tif";
		Dataset imagefile = ij.scifio().datasetIO().open( pathName );
		ImgPlus< T > img = ( ImgPlus< T > ) imagefile.getImgPlus();

		rotateAndShow( ij, img );

		double[] sigmas = { 1.5, 1.5 };
		RandomAccessibleInterval< FloatType > imgSmooth = ij.op().filter().gauss( img, sigmas );
//		Img< DoubleType > doubles = ij.op().convert().float64( Views.iterable( imgSmooth ) );

		System.out.println( Util.getTypeFromInterval( imgSmooth ).getClass() );

		FloatType maxVal = new FloatType();
		ij.op().stats().max( maxVal, Views.iterable( imgSmooth ) );
		float inverse = 1.0f / maxVal.getRealFloat();

		System.out.println( inverse );

//		System.out.println( Util.getTypeFromInterval( imgSmooth ).getClass() );

		LoopBuilder.setImages( imgSmooth ).forEachPixel( pixel -> pixel.mul( inverse ) );
		ij.ui().show( imgSmooth );
	}

	private < T extends RealType< T > > void rotateAndShow( final ImageJ ij, ImgPlus< T > img ) {
		long x = -img.dimension( 0 ) / 2;
		long y = -img.dimension( 1 ) / 2;
		AffineTransform2D transform = new AffineTransform2D();
		transform.translate( x, y );
		transform.rotate( pi / 2 );
		transform.translate( -x, -y );
		RealRandomAccessible< T > realview =
				RealViews.affineReal(
						( Views.interpolate( Views.extendBorder( img ), new NLinearInterpolatorFactory() ) ),
						transform );
		RandomAccessibleInterval< T > view = Views.interval( Views.raster( realview ), img );
		ij.ui().show( view );
	}

}

