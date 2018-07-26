package fr.jorge.facerec.objects;

import org.opencv.core.Mat;
import org.opencv.core.Point;

public class DetectedFace {

	private Mat transformedImage;
    private Mat detectedImageElement;
    private Point position;

    public DetectedFace(Mat sourceImage, Mat detectedImageElement, Point position) {

        this.transformedImage = sourceImage;
        this.detectedImageElement = detectedImageElement;
        this.position = position;
    }

    public Point getPosition() {
        return position;
    }

    public Mat getTransformedImage() {
        return transformedImage;
    }

    public Mat getDetectedImageElement() {
        return detectedImageElement;
    }

	
}
