g++ -c MatrixReaderWriter.cpp 
g++ -c -I/use/local/include panorama.cpp
g++ MatrixReaderWriter.o panorama.o -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -o panorama
