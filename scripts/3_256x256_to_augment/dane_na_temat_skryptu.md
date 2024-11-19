# Dane na temat skryptu

## Opis skryptu

Ten skrypt wykonuje augmentację obrazów JPEG, tworząc ich różnorodne wersje poprzez losowe transformacje. Użytkownik podaje ścieżki do folderu wejściowego i wyjściowego jako argumenty wiersza poleceń. Skrypt stosuje zestaw przekształceń, takich jak obrót, przesunięcie, oraz regulacja jasności i kontrastu, zdefiniowanych przy użyciu torchvision. Dla każdego obrazu generowane jest 100 wariantów z unikalnymi nazwami, które są zapisywane w folderze wyjściowym, a pasek postępu tqdm pokazuje przetwarzanie każdego pliku.

## Dodatkowe informacje

Wejściowa liczba zdjęć: 434--
Wyjściowa liczba zdjęć: 43400
Czas wykonywania się skryptu: ok. 12 godzin i 35 minut (na sprzecie opisanym powyżej)
Czas wykonywania się pojedyńczej augmentacji: ok. 1,05 sekundy
