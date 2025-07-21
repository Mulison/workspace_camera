# workspace_camera

## Implementierung der Zielerkennung

### Vorbereitung des Datasets

#### Vorbereitung der Iamge Dateien (.jpg)
#### Vorbereitung der Label Dateien (.txt)



### Erstellung der YOLO NN Reference

#### Frage
* Die Eingangsbild haben verschiedene Groß
** Lösung: entweder gleiche Groß machen, oder train nur mit den gleichgroßen Bilder

* Menge der Klassen?
** 9993506  Yellow Cone
** 9993511  Blue Cone
** 9993512  Small Orange (End)
** 9993513  Big Orange (Start)

* wie viele Layer braucht man?
* welche Layer braucht man? (full connected, flatten)
* welche Kern?
* welche Activation Function?
* padding, stride?
* learning rate, train step, epoch?
* welche Loss-Funktion? (Entropy, CrossEntropyLoss)



### Trainieren

#### Die Auswählung der Dataset, da es kein bestimmte Test-Dataset bietet



### Testen






## Implementierung der Zielverfolgung