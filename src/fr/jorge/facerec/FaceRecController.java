package fr.jorge.facerec;

import java.io.File;
import java.io.FileFilter;
import java.io.FilenameFilter;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import fr.jorge.facerec.objects.DetectedFace;
import fr.jorge.facerec.objects.RecognizedFace;
import fr.jorge.facerec.utils.Utils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class FaceRecController {
	
	//Panel Element
	@FXML
	private Button btn_camera;
	@FXML
	private Button btn_training;
	@FXML
	private Button btn_newUser;
	@FXML
	private ImageView frame_camera;
	@FXML
	private ImageView frame_training;
	@FXML
	private TextField textfield_name;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	
	private VideoCapture capture;
	private boolean cameraActive;
	private boolean isTrained = false;
	
	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;
	
	List<DetectedFace> detectedFaces;
	
	//Recognizer
	private List<Mat> images;
	private MatOfInt labelsBuffer;
	private Map<Integer, String> idToNameMapping = null;
	private FaceRecognizer faceRecognizer;
    private int[] labels;
    private double[] confidence;
    
    int counter = 0;
	
	public void init() {
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.faceCascade.load("resources/haarcascades/haarcascade_frontalface_alt.xml");		
		this.absoluteFaceSize = 0;
		
		frame_camera.setFitWidth(600);
		frame_camera.setPreserveRatio(true);
		frame_training.setFitWidth(100);
		frame_training.setPreserveRatio(true);
		
		btn_newUser.setDisable(true);
		btn_training.setDisable(true);
		this.textfield_name.setText("User 1");
	}
	
	@FXML
	public void startCamera() {
		if(!cameraActive) {
			this.capture.open(0);
			if(this.capture.isOpened()) {
				this.cameraActive = true;
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(frame_camera, imageToShow);
					}
				};
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				this.btn_camera.setText("Stop Camera");
				this.btn_newUser.setDisable(false);
			}else {
				System.err.println("Failed to open the camera connection...");
			}	
		}else {
			// the camera is not active at this point
			this.cameraActive = false;
			this.btn_camera.setText("Start Camera");			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	@FXML
	public void onTextField() {
		
	}

	@FXML
	public void startTraining(){
		train();
	}
	
	@FXML
	public void newUser() {
		Mat frame = new Mat();
		if(textfield_name.getText() != null){
			for (DetectedFace detectedElement: detectedFaces) {
				frame = detectedElement.getDetectedImageElement();
				if(detectedElement != null && detectedElement.getDetectedImageElement() != null) {
	                    Utils.saveAsJpg(frame, "resources/training/"+ textfield_name.getText() +"-"+counter+".jpg");
	                    this.btn_training.setDisable(false);
				}
				counter++;
			}
		}else {
			System.err.println("Empty TextField");
		}
		
		
	}
	
	public void train() {
		File trainingDir = new File("resources/training/");
		
		File[] imagesFiles = trainingDir.listFiles();
		idToNameMapping = new HashMap<Integer, String>();
		int idCounter = 0;
		for (File imageFile : imagesFiles) {
		    String name = imageFile.getName().split("-")[0]; //File must be on Name-Number (e.g:alex-2) form
		    if (!idToNameMapping.values().contains(name)) {
		        idToNameMapping.put(idCounter++, name);
		    }
		}
		
		images = new ArrayList<Mat>(imagesFiles.length);
		labelsBuffer = new MatOfInt(new int[imagesFiles.length]);
		
		int counter = 0;
		for (File image : imagesFiles) {

		    // reads the training image in grayscale
		    Mat img = Imgcodecs.imread(image.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
		    img = Utils.resizeFace(img);
		    frame_training.setImage(Utils.mat2Image(img));

		    // gets the id of this image
		    int labelId = idToNameMapping.keySet()
		    			  .stream()
		    			  .filter(id -> idToNameMapping.get(id).equals(image.getName().split("-")[0]))
		    			  .findFirst()
		    			  .orElse(-1);
		    // sets the image
		    images.add(img);
		    labelsBuffer.put(counter++, 0, labelId);
		    System.out.println(image.getName()+" added ! - " +labelId);
		}
		faceRecognizer = EigenFaceRecognizer.create();
		faceRecognizer.train(images, labelsBuffer);
		labels = new int[idToNameMapping.size()];
		confidence = new double[idToNameMapping.size()];
		this.isTrained = true;
	}
	/**
	 * Method for face detection and tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame){
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		detectedFaces = new ArrayList<DetectedFace>();
		
		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0){
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0){
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
				
		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		for(Rect rect : facesArray) {
			Imgproc.rectangle(frame, rect.tl(), rect.br(), new Scalar(0, 255, 255), 3);
			detectedFaces.add(new DetectedFace(frame, new Mat(frame.clone(), rect), new Point(rect.x, rect.y)));
		}
		
		if(isTrained) {
			for (DetectedFace detectedElement: detectedFaces) {
				frame = detectedElement.getTransformedImage();
				if(detectedElement != null && detectedElement.getDetectedImageElement() != null) {
					RecognizedFace recognizedFace = recognizeFace(detectedElement.getDetectedImageElement());
					String name;
					if (recognizedFace == new RecognizedFace("unknown", 0d)) {
	                    name = recognizedFace.getName();
	                }else {
	                    int percentage = (int)(100 * (1000 - recognizedFace.getConfidence()) / 1000);
	                    name = recognizedFace.getName() + " - " + percentage + "%";
	                }
					System.out.println(name);
					Point position = detectedElement.getPosition();
	                position.y -= 11;
	                position.x -= 1;
	                Imgproc.putText(frame, name, position, 2, 3, new Scalar(0, 0, 0));

	                position.y += 1;
	                position.x += 1;
	                Imgproc.putText(frame, name, position, 2, 3, new Scalar(0, 255, 255));	
				}
			}	
		}
		
	}
	
	public RecognizedFace recognizeFace(Mat face) {

        if (face == null) {
            return new RecognizedFace("unknown", 0d);
        }
        Mat resizedGrayFace = Utils.toGrayScale(Utils.resizeFace(face));
        faceRecognizer.predict(resizedGrayFace, labels, confidence);

        if (confidence[0] < 1000) {
        	System.out.println(idToNameMapping.get(labels[0]) + "- "+ confidence[0]);
            return new RecognizedFace(idToNameMapping.get(labels[0]), confidence[0]);
        }

        return new RecognizedFace("unknown", 0d);
    }
	
	/**
	 * Get a frame from the opened video stream
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame(){
		Mat frame = new Mat();
		
		if (this.capture.isOpened()){
			try{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty()){
					// face detection
					this.detectAndDisplay(frame);
				}
			}catch (Exception e){
				System.err.println("Image elaboration failure: " + e);
			}
		}
		return frame;
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition(){
		if (this.timer!=null && !this.timer.isShutdown()){
			try{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
				File trainingDir = new File("resources/training/");
				File[] imagesFiles = trainingDir.listFiles();
				for(File image: imagesFiles) {
					image.delete();
				}
			}catch (InterruptedException e){
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		if (this.capture.isOpened()){
			// release camera
			this.capture.release();
		}
	}
	
	//Updating imageview in JavaFX main thread
	private void updateImageView(ImageView view, Image image){
		Utils.onFXThread(view.imageProperty(), image);
	}
}
