const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const inputFolder = path.join(__dirname, "../../photos/crop-and-black");
const outputFolder = path.join(__dirname, "../../photos/256x256");
if (!fs.existsSync(outputFolder)) {
  fs.mkdirSync(outputFolder);
}

// Odczytaj wszystkie pliki w folderze
fs.readdir(inputFolder, (err, files) => {
  if (err) {
    console.error("Błąd przy odczytywaniu folderu:", err);
    return;
  }

  // Przetwarzaj każdy plik w folderze
  files.forEach(async (file) => {
    const inputImagePath = path.join(inputFolder, file);
    const outputImagePath = path.join(outputFolder, `${file}`);

    // Sprawdź, czy plik jest obrazem (np. z rozszerzeniem .jpg, .jpeg, .png)
    if (/\.(png)$/i.test(file)) {
      try {
        // Wczytaj obraz i pobierz jego wymiary
        const { width, height } = await sharp(inputImagePath).metadata();

        // Oblicz środek obrazu i ustal współrzędne do wycięcia 256x256
        const left = Math.floor((width - 256) / 2);
        const top = Math.floor((height - 256) / 2);

        // Wytnij środkowy obszar 256x256 i zapisz wynikowy obraz
        await sharp(inputImagePath)
          .extract({ width: 256, height: 256, left: left, top: top })
          .toFile(outputImagePath);
      } catch (error) {
        console.error(
          `Błąd podczas przetwarzania obrazu ${inputImagePath}:`,
          error
        );
      }
    } else {
      throw Error(`Plik ${inputImagePath} nie jest obrazem PNG`);
    }
  });
});
