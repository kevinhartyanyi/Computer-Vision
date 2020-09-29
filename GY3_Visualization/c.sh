g++ -c MatrixReaderWriter.cpp 
g++ -c FV.cpp
g++ MatrixReaderWriter.o FV.o -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -o FV
