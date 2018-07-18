package com.mycompany.imagej;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.algorithm.neighborhood.Neighborhood;
import net.imglib2.algorithm.neighborhood.RectangleShape;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.AffineTransform2D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class Utilities {

	public static < T extends RealType< T > > Map< Integer, List > peakLocalMax(
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

	public static < T extends RealType< T > > RandomAccessibleInterval< T > rotate(
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
