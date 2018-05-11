1，	Télécharger le opencv avec brew : brew install opencv
（vous pouvez le consulter par le path: /usr/local/Cellar/opencv）

2，  Télécharger le pkg-config avec brew : brew install pkg-config



3,   command de complier: $ g++ @@@.cpp ``pkg-config --libs --cflags opencv`` -o ### -framework OpenCL

4,   command d'execution: $ ./### ***.jpg
