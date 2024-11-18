# Dane na temat skryptu

## Opis skryptu

Ten skrypt używa biblioteki sharp do przycinania obrazów PNG do rozmiaru 256x256 pikseli, koncentrując się na centralnym obszarze obrazu. Obrazy są pobierane z określonego folderu wejściowego, a wynikowe przycięte obrazy są zapisywane w folderze wyjściowym. Skrypt najpierw sprawdza istnienie folderu wyjściowego i tworzy go, jeśli nie istnieje. Następnie odczytuje wszystkie pliki PNG w folderze wejściowym, sprawdza ich wymiary, wyznacza środek obrazu i przycina centralną część o wymiarach 256x256 pikseli. W razie problemów z przetwarzaniem obrazów lub jeśli plik nie jest PNG, skrypt zgłasza odpowiedni błąd.

## Dodatkowe informacje

Wejściowa liczba zdjęć: 43400
Wyjściowa liczba zdjęć: 43400
Czas wykonywania się skryptu: 
Czas wykonywania się pojedynczego przycięcia: 