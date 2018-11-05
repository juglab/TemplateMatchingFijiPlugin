package com.mycompany.imagej;

import java.util.ArrayList;
import java.util.List;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.Point;
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

	public static < T extends RealType< T > > ArrayList< Point > peakLocalMax(
			RandomAccessibleInterval< T > source,
			int radius ) {

		ArrayList< Point > localMaxCoords = new ArrayList<>();
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

				Point coords = new Point( center.numDimensions() );
				coords.setPosition( center );
				localMaxCoords.add( coords );
			}
		}

		return localMaxCoords;
	}

	public static < T extends RealType< T > > RandomAccessibleInterval< T > rotate(
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
		return Views.interval( Views.raster( realview ), template );
	}
}

