1，	Télécharger le opencv avec brew : brew install opencv
（vous pouvez le consulter par le path: /usr/local/Cellar/opencv）

2，  Télécharger le pkg-config avec brew : brew install pkg-config

3,   command de complier: $ g++ @@@.cpp `pkg-config --libs --cflags opencv` -o ### -framework OpenCL

4,   command d'execution: $ ./### ***.jpg

5,    command d'execution: $ ./### picture1/ ***

le premier parametre est le dossiser d'images , je le ajoute juste 1. vous pouvez l'ajouter plusieur images.

le deuxième parametre est le nom de test du temp.Il y a trois paramètres: Si vous souhaitez afficher l'heure de lecture du fichier, entrez "lecture" Si vous souhaitez afficher l'heure de calcul, entrez "calcul" Si vous souhaitez afficher l'heure de transmission, entrez "trans".

Pour tester le temps de fonctionnement de l'ensemble du programme, vous devez utiliser la commande "time".

par example : $ ./main picture1/ lecture
