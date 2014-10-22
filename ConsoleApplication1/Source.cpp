// Facial Detection Project
// Serial Implementation
// Tyler Apgar and Amadeus Sanchez
// 228 seconds elapsed.
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	const char filenameInput[] = "filenameInput.txt";
	const char haarcascadeFilename[] = "haarcascade_frontalface_alt.xml";
	const char outputImageSignature[] = "_output";
	const char imageExtension[] = ".jpg";
	//create the cascade classifier object used for the face detection
	CascadeClassifier face_cascade;
	//use the haarcascade_frontalface_alt.xml library
	face_cascade.load(haarcascadeFilename);

	string filename;
	ifstream myfile(filenameInput);
	if (myfile.is_open())
	{
		time_t start, end;
		time(&start);
		while (getline(myfile, filename))
		{
			//setup image file used in the capture process
			Mat captureFrame;
			Mat grayscaleFrame;
				captureFrame = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
				

			// convert captured image to gray scale and equalize it
			// so that it can boost the detection rate
			cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
			equalizeHist(grayscaleFrame, grayscaleFrame);

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
			outputFilename+=filename.substr(0, indexExtension);
			outputFilename += outputImageSignature;
			outputFilename += imageExtension;
			imwrite(outputFilename.c_str(), captureFrame);
		}
		myfile.close();
		time(&end);
		double dif = difftime(end, start);
		printf("Elasped time is %.2lf seconds.", dif);
	}

	else cout << "Unable to open file";

		
	return 0;
}