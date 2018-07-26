package fr.jorge.facerec.objects;

public class RecognizedFace {
    private final String name;
    private final Double confidence;

    public RecognizedFace(String name, Double confidence) {
        this.confidence = confidence;
        this.name = name;
    }

    public Double getConfidence() {
        return confidence;
    }

    public String getName() {
        return name;
    }
}
