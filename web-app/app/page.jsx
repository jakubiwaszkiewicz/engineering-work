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
import { Progress } from "../components/ui/progress";
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

  const classifyImage = async () => {
    setIsClassifying(true);
    // Simulating classification process
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setClassificationResult(
      `Classification result using ${selectedModel}: Sample Class`
    );
    setIsClassifying(false);
  };

  const deleteImage = () => {
    setImage(null);
    setClassificationResult(null);
    if (typeof window !== "undefined") {
      localStorage.removeItem("classificationImage");
    }
  };

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-4xl font-bold mb-8 text-center text-primary">
        Klasyfikacja ramki pszczelej
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
              className="max-w-full h-auto rounded-lg shadow-lg mb-4"
            />
            <Button variant="destructive" onClick={deleteImage} className="">
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
              "Maszyna Wektorów Nośnych (SVM)",
              "Splotowa Sieć Neuronowa - LesNet-5 (CNN)",
            ].map((model) => (
              <Button
                key={model}
                variant={selectedModel === model ? "default" : "outline"}
                onClick={() => setSelectedModel(model)}
                className="flex-1"
              >
                {model.charAt(0).toUpperCase() + model.slice(1)}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      <Button
        onClick={classifyImage}
        disabled={!image || isClassifying}
        className="w-full mb-8"
      >
        {isClassifying ? "Klasyfikacja..." : "Sklasyfikuj zdjęcie"}
      </Button>

      {isClassifying && <Progress value={66} className="mb-8" />}

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
