package fr.jorge.facerec;

/**
 * Created by ALexandre JORGE on 26/07/2018
 * 
 * For 
 */
	
import org.opencv.core.Core;

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;


public class Main extends Application {
	@Override
	public void start(Stage primaryStage) {
		try {
			FXMLLoader loader = new FXMLLoader(getClass().getResource("FaceRec.fxml"));
			BorderPane root = loader.load();
			Scene scene = new Scene(root,1000,400);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			primaryStage.setTitle("FaceRec");
			primaryStage.setScene(scene);
			primaryStage.show();
			
			FaceRecController controller = loader.getController();
			controller.init();
			
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		launch(args);
	}
}
