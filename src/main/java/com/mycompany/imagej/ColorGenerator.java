package com.mycompany.imagej;

import net.imglib2.type.numeric.ARGBType;

import java.util.Arrays;
import java.util.List;

public class ColorGenerator
{
	private static List< Integer > color = Arrays.asList(
			0xFF00FF00,
			0xFFFF0000,
			0xFFFFFF00,
			0xFFFF00FF,
			0xff803e75,
			0xffff6800,
			0xffa6bdd7,
			0xffc10020,
			0xffcea262,
			0xff817066,
			0xff007d34,
			0xfff6768e,
			0xff00538a,
			0xffff7a5c,
			0xff53377a,
			0xffff8e00,
			0xffb32851,
			0xfff4c800,
			0xff7f180d,
			0xff93aa00,
			0xff593315,
			0xfff13a13,
			0xff232c16
	);

	public static ARGBType getColor( int count )
	{
		return new ARGBType( color.get( count % color.size() ) );
	}
}
