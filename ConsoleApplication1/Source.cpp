// Facial Detection Project
// Serial Implementation
// Tyler Apgar and Amadeus Sanchez
// 228 seconds elapsed.
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <omp.h>

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

bool Done;	// Global flag used by the threads to communicate when all tasks are done!

std::string getNextTask(ifstream &fin)
{
	std::string filename = "";
#pragma omp critical			// Critical section to access variables: fin, n, Done 
	{
		if (!Done) {
			getline(fin, filename);
			if (filename.empty()) Done = true;
		}
	}
	return filename;
}

int main(int argc, const char** argv)
{
	const char filenameInput[] = "filenameInput.txt";
	const char haarcascadeFilename[] = "haarcascade_frontalface_alt.xml";
	const char outputImageSignature[] = "_output";
	const char imageExtension[] = ".jpg";
	//create the cascade classifier object used for the face detection
	

	int desiredNumThreads;
	ifstream fin(filenameInput);
	
	if (fin.is_open())
	{
		cout << "How many threads? "; cin >> desiredNumThreads;
		Done = false;
		double startTime = omp_get_wtime();

#pragma omp parallel num_threads(desiredNumThreads)
		{
			std::string filename = getNextTask(fin);	// If there are no more tasks, sets n to 0 AND sets Done to true
			while (!Done) {
				//setup image file used in the capture process
				CascadeClassifier face_cascade;
				//use the haarcascade_frontalface_alt.xml library
				face_cascade.load(haarcascadeFilename);
				Mat captureFrame;
				Mat grayscaleFrame;
				captureFrame = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);


				// convert captured image to gray scale and equalize it
				// so that it can boost the detection rate
				cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
				//create a vector array to store the face found
				std::vector<Rect> faces;

				//find faces and store them in the vector array
				face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_SCALE_IMAGE, Size(20, 20));
				cout << faces.size() << " faces were found" << endl;

				//draw a rectangle for all found faces in the vector array on the original image
				for (int i = 0; i < faces.size(); i++)
				{
					Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
					Point pt2(faces[i].x, faces[i].y);

					rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
				}

				//print the output
				string outputFilename;
				int indexExtension = filename.find(".jpg");
				outputFilename += filename.substr(0, indexExtension);
				outputFilename += outputImageSignature;
				outputFilename += imageExtension;
				imwrite(outputFilename.c_str(), captureFrame);


				filename = getNextTask(fin);	// If there are no more tasks, sets n to 0 AND sets Done to true
			}
			fin.close();
			}
		double endTime = omp_get_wtime();
		cout << "Completing all tasks using " << desiredNumThreads << " required " << (endTime - startTime) << " seconds." << endl;

		
	}
	else{
		cout << "Unable to open file";
	}
	
		
	return 0;
}