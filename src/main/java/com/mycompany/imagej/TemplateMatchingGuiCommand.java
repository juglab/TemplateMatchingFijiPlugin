package com.mycompany.imagej;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.util.Util;
import net.miginfocom.swing.MigLayout;
import org.scijava.Context;
import org.scijava.command.Command;
import org.scijava.log.Logger;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;
import org.scijava.ui.behaviour.util.RunnableAction;

import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.List;

@Plugin( menuPath = "Plugins>Segmentation>Template Matching Segmentation (GUI)", type = Command.class )
public class TemplateMatchingGuiCommand implements Command
{
	@Parameter
	Dataset image;

	@Parameter
	Context context;

	@Parameter
	Logger logger;

	@Parameter
	UIService ui;

	JFrame frame;

	private TemplateMatchingPanel panel;

	@Override
	public void run()
	{
		panel = new TemplateMatchingPanel(toDoubleType(image.getImgPlus().getImg()), context, logger);
		frame = new JFrame("Template Matching Segmentation");
		frame.setLayout( new MigLayout("", "[grow]", "[][]") );
		frame.add( panel.getPanel(), "grow, wrap");
		frame.addWindowListener( new WindowAdapter()
		{
			@Override
			public void windowClosed( WindowEvent windowEvent )
			{
				panel.close();
			}
		} );
		frame.add( newButton( "Show Results", this::showResults ), "split 2");
		frame.add( newButton( "Save Results", this::saveResults ) );
		frame.pack();
		frame.setSize( 500, 500 );
		frame.setVisible( true );
	}

	private void showResults()
	{
		List< RandomAccessibleInterval< IntType > > results = panel.getOutputs();
		results.forEach( ui::show );
	}

	private void saveResults()
	{
		List< RandomAccessibleInterval< IntType > > results = panel.getOutputs();
		JFileChooser chooser = new JFileChooser();
		chooser.setCurrentDirectory(new java.io.File("."));
		chooser.setDialogTitle("Select Directory to Save Result Images.");
		chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		chooser.setAcceptAllFileFilterUsed(false);
		if( chooser.showSaveDialog( frame ) == JFileChooser.APPROVE_OPTION ) {
			TemplateMatchingPlugin.saveImages(results, chooser.getSelectedFile() );
		}
	}


	private JButton newButton( String title, Runnable action )
	{
		final JButton button = new JButton( title );
		button.addActionListener( a -> action.run() );
		return button;
	}

	private RandomAccessibleInterval<DoubleType> toDoubleType( RandomAccessibleInterval< ? extends RealType< ? > > image )
	{
		if( Util.getTypeFromInterval(image) instanceof DoubleType )
			return (RandomAccessibleInterval<DoubleType>) image;
		return Converters.convert( image, (i, o) -> o.setReal( i.getRealDouble() ), new DoubleType(  ) );
	}

	public static void main(String... args) {
		ImageJ imageJ = new ImageJ();
		imageJ.ui().showUI();
		imageJ.ui().show( ArrayImgs.unsignedBytes( new byte[] { 0, -1 }, 1, 2 ));
		imageJ.command().run( TemplateMatchingGuiCommand.class, true );
	}
}
