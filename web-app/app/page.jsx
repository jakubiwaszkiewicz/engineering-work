"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import imageCompression from "browser-image-compression";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { Upload, Image as ImageIcon, AlertTriangle } from "lucide-react";

export default function ImageClassification() {
  const [image, setImage] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [selectedModel, setSelectedModel] = useState("model1");
  const [isClassifying, setIsClassifying] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      try {
        const compressedFile = await imageCompression(file, {
          maxSizeMB: 1,
          maxWidthOrHeight: 1024,
          useWebWorker: true,
        });

        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target?.result) {
            setImage(e.target.result);
            localStorage.setItem("classificationImage", e.target.result);
          }
        };
        reader.readAsDataURL(compressedFile);
      } catch (error) {
        console.error("Error compressing image:", error);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
  });

  const classifyImage = async (model) => {
    setIsClassifying(true);
    setClassificationResult(null);

    let chosenModel;
    if (model == "GNB") {
      chosenModel = "gnb";
    } else if (model == "SVM") {
      chosenModel = "sgdc";
    } else {
      chosenModel = "lenet";
    }
    console.log("Sending image to the server for classification...");
    console.log(`http://127.0.0.1:5000/${chosenModel}`);
    try {
      const response = await fetch(`http://127.0.0.1:5000/${chosenModel}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_base64: image.split(",")[1],
        }),
      });

      const result = await response.json();
      if (response.ok) {
        console.log("Classification successful:", result);
        setClassificationResult(
          `Zasklepienie: ${result.Prediction.Cap}, twardość: ${result.Prediction.Hardness}, miód: ${result.Prediction.Honey}`
        );
      } else {
        console.error("Classification failed:", result.error);
        setClassificationResult("Error during classification: " + result.error);
      }
    } catch (error) {
      console.error("Error connecting to the server:", error);
      setClassificationResult("Error connecting to the server");
    } finally {
      console.log("Classification completed");
      setIsClassifying(false);
    }
  };

  const deleteImage = () => {
    setImage(null);
    localStorage.removeItem("classificationImage");
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center text-primary">
        Klasyfikator Ramek Pszczelarskich
      </h1>
      <Card>
        <CardHeader>
          <CardTitle>Załaduj zdjęcie</CardTitle>
          <CardDescription>
            Przeciągnij tutaj zdjęcie, lub klinij by wybrać
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className="border-2 border-dashed border-primary/50 rounded-lg p-12 text-center cursor-pointer hover:bg-primary/5 transition-colors"
          >
            <input {...getInputProps()} />
            {isDragActive ? (
              <p className="text-primary">Przeciagnij zdjęcie tutaj...</p>
            ) : (
              <div>
                <Upload className="mx-auto h-12 w-12 text-primary mb-4" />
                <p className="text-primary">
                  Przeciągnij zdjęcie tutaj, albo kliknij by wybrać
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {image && (
        <Card className="mt-8 mb-8">
          <CardHeader>
            <CardTitle>Załadowane zdjęcie</CardTitle>
          </CardHeader>
          <CardContent>
            <img
              src={image}
              alt="Zaladowane zdjecie"
              className="max-w-full mx-auto h-auto rounded-lg shadow-lg mb-4"
            />
            <Button
              variant="destructive"
              onClick={deleteImage}
              className="w-full"
            >
              Usuń zdjęcie
            </Button>
          </CardContent>
        </Card>
      )}

      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Wybierz model do klasyfikacji</CardTitle>
          <CardDescription>
            Wybierz jeden z trzech dostępnych modeli do klasyfikacji zdjęcia
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            {[
              "Gaussowski Naiwny Bayes (GNB)",
              "Maszyna Wektorów Nośnych z uczeniem SGD (SVM)",
              "Splotowa Sieć Neuronowa - LesNet-5 (CNN)",
            ].map((model) => (
              <Button
                key={model}
                variant={selectedModel === model ? "default" : "outline"}
                onClick={() => {
                  setSelectedModel(model);
                }}
                className="flex-1"
              >
                {model.charAt(0).toUpperCase() + model.slice(1)}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Button
        onClick={() => {
          if (selectedModel == "Gaussowski Naiwny Bayes (GNB)") {
            classifyImage("GNB");
          } else if (
            selectedModel == "Maszyna Wektorów Nośnych z uczeniem SGD (SVM)"
          ) {
            classifyImage("SVM");
          } else {
            classifyImage("CNN");
          }
        }}
        className="w-full mb-8"
      >
        Sklasyfikuj zdjęcie
      </Button>

      {classificationResult && (
        <Alert>
          <ImageIcon className="h-4 w-4" />
          <AlertTitle>Wynik Klasyfikacji</AlertTitle>
          <AlertDescription>{classificationResult}</AlertDescription>
        </Alert>
      )}

      {!image && !isClassifying && (
        <Alert variant="destructive" className="mt-8">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Nie załadowano żadnego zdjęcia</AlertTitle>
          <AlertDescription>
            Aby rozpocząć, załaduj zdjęcie do klasyfikacji
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
