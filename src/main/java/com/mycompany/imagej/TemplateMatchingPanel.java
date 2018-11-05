/**
 *
 */

package com.mycompany.imagej;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import javax.swing.BorderFactory;
import javax.swing.DefaultListModel;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingWorker;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import com.mycompany.imagej.ColorGenerator;
import com.mycompany.imagej.TemplateMatchingAlgorithm;
import ij.IJ;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import org.scijava.Context;
import org.scijava.command.CommandService;
import org.scijava.log.Logger;
import org.scijava.plugin.Parameter;

import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvHandlePanel;
import bdv.util.BdvStackSource;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.miginfocom.swing.MigLayout;


/**
 * @author Mangal Prakash
 */
public class TemplateMatchingPanel implements AutoCloseable {

	@Parameter
	private Context context;

	@Parameter
	private CommandService commandService;

	public Logger log;

	DefaultListModel< String > model = new DefaultListModel<>();
	private final JList< String > listTemplates = new JList<>( model );

	private final List< Boolean > listSegmenationPerformedWithTemplateIndicator = new ArrayList< Boolean >();

	private JTextField threshold;

	private JTextField segRad;

	private BdvHandlePanel bdv;

	private ArrayList< BdvStackSource< IntType > > overlayObjectList = null;

	private ArrayList< List< RandomAccessibleInterval< IntType > > > listofOutputsForAllTemplates =
			new ArrayList< List< RandomAccessibleInterval< IntType > > >();

	private boolean templateRunMoreThanOnce = false;

	private RandomAccessibleInterval< DoubleType > rawData;

	public TemplateMatchingPanel( RandomAccessibleInterval< DoubleType > image, Context context, Logger log ) {
		this.rawData = image;
		this.log = log;
		context.inject( this );
	}

	public JPanel getPanel() {
		final JPanel controls = initControlsPanel();
		bdv = initBdv( rawData );
		return wrapToJPanel( initSplitPane( controls, bdv.getViewerPanel() ) );
	}

	private JSplitPane initSplitPane( JPanel left, JPanel right )
	{
		final JSplitPane splitPane = new JSplitPane( JSplitPane.HORIZONTAL_SPLIT, left, right );
		splitPane.setOneTouchExpandable( true );
		splitPane.setDividerLocation( 300 );
		return splitPane;
	}

	private < T > BdvHandlePanel initBdv( RandomAccessibleInterval< T > img )
	{
		final BdvHandlePanel bdv = new BdvHandlePanel( null, Bdv.options().is2D() );
		BdvFunctions.show( img, "img", Bdv.options().addTo( bdv ) );
		return bdv;
	}

	private JPanel wrapToJPanel( JSplitPane splitPane )
	{
		final JPanel splittedPanel = new JPanel();
		splittedPanel.setLayout( new BorderLayout() );
		splittedPanel.add( splitPane, BorderLayout.CENTER );
		return splittedPanel;
	}

	private JPanel initControlsPanel()
	{
		final MigLayout layout = new MigLayout( "fill", "[grow]", "" );
		final JPanel controls = new JPanel( layout );

		final JPanel list = initListPanel();
		controls.add( list, "h 100%, grow, wrap" );

		JPanel helper = initHelperPanel();
		controls.add( helper, "growx, wrap" );

		JButton bStartSegmentation = new JButton( "start matching with selected template" );
		bStartSegmentation.addActionListener( this::onStartSegmentationButtonClicked );
		controls.add( bStartSegmentation, "growx, gapy 5 0, wrap" );
		return controls;
	}


	private JPanel initHelperPanel()
	{

		JPanel helper = new JPanel( new MigLayout( "" ) );
		helper.add( new JLabel( "Threshold:" ) );
		threshold = new JTextField();
		threshold.setText( "0.3" );
		ImageIcon imageIcon = createQuestionIcon();
		JLabel thresholdIconLabel = new JLabel( imageIcon );
		thresholdIconLabel.setToolTipText( "Threshold for template matching" );
		helper.add( threshold, "width 100:20" );
		helper.add( thresholdIconLabel, "wrap" );
		helper.add( new JLabel( "Marker Radius:" ), "" );
		segRad = new JTextField();
		segRad.setText( "3" );
		JLabel segRadiusIconLabel = new JLabel( imageIcon );
		segRadiusIconLabel.setToolTipText( "Radius of circle to draw detection points after matching is performed" );
		helper.add( segRad, "width 100:20" );
		helper.add( segRadiusIconLabel );
		return helper;
	}

	private ImageIcon createQuestionIcon() {
		ImageIcon imageIcon = new ImageIcon( this.getClass().getResource( "/questionIcon.gif" ) );
		Image image = imageIcon.getImage();
		Image newimg = image.getScaledInstance( 30, 30, java.awt.Image.SCALE_SMOOTH );
		imageIcon = new ImageIcon( newimg );
		return imageIcon;
	}


	private JPanel initListPanel()
	{
		final JButton bAdd = initAddButton();
		final JButton bRemove = initRemoveButton();
		final JPanel list = new JPanel( new BorderLayout() );
		list.add( listTemplates, BorderLayout.CENTER );
		listTemplateSelectionListenerInit();
		list.setBorder( BorderFactory.createTitledBorder( "Templates" ) );
		JPanel helper = new JPanel( new FlowLayout( FlowLayout.RIGHT ) );

		helper.add( bAdd );
		helper.add( bRemove );
		list.add( helper, BorderLayout.SOUTH );

		JScrollPane scrollPane = new JScrollPane( listTemplates );
		list.add( scrollPane, BorderLayout.CENTER );
		return list;
	}

	private void listTemplateSelectionListenerInit() {
		listTemplates.addListSelectionListener( new ListSelectionListener() {

			@Override
			public void valueChanged( ListSelectionEvent e ) {
				if ( !e.getValueIsAdjusting() ) {
					if ( listTemplates.getModel().getSize() > 0 ) {
						if ( listTemplates.getSelectedIndex() == -1 ) {
							listTemplates.setSelectedIndex( 0 );

						}
						if ( listSegmenationPerformedWithTemplateIndicator.get( listTemplates.getSelectedIndex() ) ) {
							clearOverlayListAndBdvOverlay();
							showOverlay( listofOutputsForAllTemplates.get( listTemplates.getSelectedIndex() ) );
						} else {
							clearOverlayListAndBdvOverlay();
						}
					}

				}

			}
		} );
	}


	private JButton initRemoveButton()
	{
		final JButton bRemove = new JButton( "-" );
		bRemove.addActionListener( this::onRemoveButtonClicked );
		return bRemove;
	}

	private JButton initAddButton()
	{
		final JButton bAdd = new JButton( "+" );
		bAdd.addActionListener( this::onAddButtonClicked );
		return bAdd;
	}


	public List< RandomAccessibleInterval< IntType > > fetchOutputs() {
		if ( listSegmenationPerformedWithTemplateIndicator.get( listTemplates.getSelectedIndex() ) ) {
			List< RandomAccessibleInterval< IntType > > segOutputs = listofOutputsForAllTemplates.get( listTemplates.getSelectedIndex() );
			templateRunMoreThanOnce = true;
			return segOutputs;
		} else {
			final File file = new File( listTemplates.getSelectedValue() );
			RandomAccessibleInterval< DoubleType > template = (Img< DoubleType > ) ImagePlusAdapter.wrapReal( IJ.openImage( file.getAbsolutePath() ) );
			Double matchingThreshold = Double.parseDouble( threshold.getText() );
			int segmentationRadius = Integer.parseInt( segRad.getText() );
			List< RandomAccessibleInterval< IntType > > segOutputs = ( List )
					new TemplateMatchingAlgorithm( context ).calculate( rawData, template, segmentationRadius, matchingThreshold, null );
			listSegmenationPerformedWithTemplateIndicator.set( listTemplates.getSelectedIndex(), true );
			return segOutputs;
		}

	}

	private void dialogWaiting() {
		JProgressBar progressBar = new JProgressBar();
		progressBar.setIndeterminate( true );
		JTextArea msgLabel;
		final JDialog dialogWaiting;
		JPanel panel;

		msgLabel = new JTextArea( "Matching with selected template and creating segmentations..." );
		msgLabel.setEditable( false );

		panel = new JPanel( new BorderLayout( 5, 5 ) );
		panel.add( msgLabel, BorderLayout.PAGE_START );
		panel.add( progressBar, BorderLayout.CENTER );
		panel.setBorder( BorderFactory.createEmptyBorder( 11, 11, 11, 11 ) );

		dialogWaiting = new JDialog( ( JFrame ) null, "Progress...", true );
		dialogWaiting.getContentPane().add( panel );
		dialogWaiting.setResizable( false );
		dialogWaiting.pack();
		dialogWaiting.setSize( 450, dialogWaiting.getHeight() );
		dialogWaiting.setLocationRelativeTo( null );
		dialogWaiting.setDefaultCloseOperation( JDialog.DO_NOTHING_ON_CLOSE );
		dialogWaiting.setAlwaysOnTop( true );
		msgLabel.setBackground( panel.getBackground() );

		fetchOutputsWhileWaitingDialogOn( dialogWaiting );
	}

	private void fetchOutputsWhileWaitingDialogOn( final JDialog dialogWaiting ) {
		SwingWorker worker = new SwingWorker() {

			private List< RandomAccessibleInterval< IntType > > fetchedOutputs;

			@Override
			protected void done() {

				dialogWaiting.dispose();
				if ( templateRunMoreThanOnce == false ) {
					listofOutputsForAllTemplates.add( fetchedOutputs );
				}
				showOverlay( fetchedOutputs );

				System.out.println( listofOutputsForAllTemplates.size() );
			}

			@Override
			protected List< RandomAccessibleInterval< IntType > > doInBackground() {
				fetchedOutputs = fetchOutputs();
				return fetchedOutputs;
			}
		};

		worker.execute();
		dialogWaiting.setVisible( true );
	}


	public List< RandomAccessibleInterval< IntType > > getOutputs() {
		List< RandomAccessibleInterval< IntType > > segmentationMasksForAllTemplates =
				listofOutputsForAllTemplates
						.stream()
						.flatMap( List::stream )
						.collect( Collectors.toList() );
		return segmentationMasksForAllTemplates;

	}

	public void setLogger(Logger logger) {
		log = logger;
	}

	@Override
	public void close() {
		bdv.close();
	}

	public void onAddButtonClicked( ActionEvent e ) {
		JFileChooser fileChooser = new JFileChooser();
		int returnName = fileChooser.showOpenDialog( null );
		String path;

		if ( returnName == JFileChooser.APPROVE_OPTION ) {
			File f = fileChooser.getSelectedFile();
			if ( f != null ) { // Make sure the user didn't choose a directory.

				path = f.getAbsolutePath();
				System.out.println( path );
				listSegmenationPerformedWithTemplateIndicator.add( false );
				model.addElement( path );
			}
		}
	}

	public void onRemoveButtonClicked( ActionEvent e ) {
		int idx = listTemplates.getSelectedIndex();
		model.remove( idx );
		listSegmenationPerformedWithTemplateIndicator.remove( idx );
		if ( idx < listofOutputsForAllTemplates.size() ) {
			listofOutputsForAllTemplates.remove( idx );
		}
	}


	public void onStartSegmentationButtonClicked( ActionEvent e ) {

		clearOverlayListAndBdvOverlay();
		dialogWaiting();

	}

	private void showOverlay( List< RandomAccessibleInterval< IntType > > outputs ) {
		int overlayBucketSize = outputs.size();
		overlayObjectList = new ArrayList< BdvStackSource< IntType > >( overlayBucketSize );
		int count = -1;

		for ( RandomAccessibleInterval< IntType > output : outputs ) {
			BdvStackSource< IntType > entry = BdvFunctions.show( output, "Overlays", Bdv.options().addTo( bdv ) );
			overlayObjectList.add( entry );
			count += 1;
			entry.setColor( ColorGenerator.getColor( count ) );
			entry.setDisplayRange( 0, 0 );
		}
	}

	private void clearOverlayListAndBdvOverlay() {

		if ( overlayObjectList != null ) {
			for ( BdvStackSource< IntType > overlayObject : overlayObjectList ) {
				overlayObject.removeFromBdv();
			}
			overlayObjectList.clear();
		}

	}


}
