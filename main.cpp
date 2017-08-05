#include "mainwindow.h"
#include <QApplication>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cmshapecontext.h"

using namespace dlib;
using namespace std;
using namespace cv;

typedef enum faceType{
    ROUND,
    LONG,
    TRIANGLE,
    HEART,
    SQUARE
}faceType;

const int N_IMAGES = 15;
const int candidate_points = 64;
const int LANDS = 17;
const int IM_PER_CLASS = 3;
int mul = 1;

Mat getSilhouette(frontal_face_detector detector, shape_predictor sp, string fileName, std::vector<Point2d>& landmarks);
Mat getSilhouetteDB(frontal_face_detector detector, shape_predictor sp, string fileName);

std::vector<Point2d> getContour(Mat silhouette);
std::vector<complex<float>> getFD(std::vector<Point2d> c1, Mat imc1);
double compareFD(std::vector<complex<float>> fd1, std::vector<complex<float>> fd2);
void extractAverageColour(std::vector<Point2d> contour, Mat inputIM);
void registerFD(std::vector<complex<float>> imDBFD, faceType fType, ofstream& DBFDData);
faceType getCategory(int index);
std::vector<std::vector<complex<float>>> loadDB();


int main(int argc, char *argv[])
{
    std::vector< tuple<std::vector<complex<float>> , faceType > > FDDataset;
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    CmShapeContext shapeContextEvaluator;
    Mat imDB,myIM, inputIM;
    std::vector<Point2d> imDBpoints,myIMpoints, myLandmarks;
    std::vector<complex<float>> imDBFD, myIMFD;
    double score, sum;
    int c;
    ofstream DBFDData;
    std::vector<std::vector<complex<float>>> dataFromDB;

    deserialize("/home/isadora/Desktop/shape_predictor_68_face_landmarks.dat") >> sp;

    //LOAD IMAGES FROM THE DATASET AND COMPUTE FDs
    inputIM = imread("/home/isadora/Desktop/finaltest/DBimgs/imtest1.jpg");
    myIM = getSilhouette(detector, sp, "/home/isadora/Desktop/finaltest/DBimgs/imtest1.jpg", myLandmarks);
    myIMpoints = getContour(myIM);
    //myIMFD = getFD(myIMpoints, myIM);


    //VISUALISE LANDMARKS
//    Mat color;
//    cv::cvtColor(myIM, color, cv::COLOR_GRAY2BGR);
//    for(int j = 0; j < myIMpoints.size(); j++){
//        Point p = Point (myIMpoints.at(j).x,myIMpoints.at(j).y);
//        circle(color, p, 1, Scalar(0,0,255),1,8,0);
//    }
    //imwrite( "/home/isadora/Desktop/finaltest/DBimgs/Gray_Image.jpg", color );


    //GET FD INFO FROM DB AND REGISTER
//    DBFDData.open ("/home/isadora/Desktop/finaltest/DBimgs/DBFDData.txt");
//    for(unsigned int i = 0; i < N_IMAGES; i++){
//        imDB = getSilhouetteDB(detector, sp, string("/home/isadora/Desktop/finaltest/DBimgs/imtest")+ to_string(i) + ".jpg");
//        imDBpoints = getContour(imDB);
//        imDBFD = getFD(imDBpoints, imDB);
//        //Comment next line to skip registration to file
//        registerFD(imDBFD, getCategory(i), DBFDData);

//        //SCORES AND DISPLAY SCORES
//        //score = shapeContextEvaluator.matchCost(imDBpoints, myIMpoints);
//        score = compareFD(myIMFD, imDBFD);
//        cout << "# " << i << " diff " << score << endl;

//        if (c==3){sum = 0.0; c = 0;}
//        sum += score;
//        if (c==2)cout << "SUM " << sum << endl;
//        c++;
//    }
//    DBFDData.close();


    //compare FD with DB from file
    dataFromDB = loadDB;


    //COLOUR EXTRACTION
    //face square under eyes uplips to avoid eyecolour and
//    std::vector<Point2d> faceColourSelection;
//    faceColourSelection.push_back(myLandmarks.at(1));
//    faceColourSelection.push_back(myLandmarks.at(2));
//    faceColourSelection.push_back(myLandmarks.at(3));
//    faceColourSelection.push_back(myLandmarks.at(13));
//    faceColourSelection.push_back(myLandmarks.at(14));
//    faceColourSelection.push_back(myLandmarks.at(15));
//    extractAverageColour(faceColourSelection, inputIM);
//    //Mouth
//    std::vector<Point2d> mouthColourSelection;
//    for(unsigned int i = 48; i < 59; i++) mouthColourSelection.push_back(myLandmarks.at(i));
//    extractAverageColour(mouthColourSelection, inputIM);

//    //Eye (just one)
//    std::vector<Point2d> eyeColourSelection;
//    for(unsigned int i = 36; i < 41; i++) eyeColourSelection.push_back(myLandmarks.at(i));
}

std::vector<std::vector<complex<float>>> loadDB(){
    std::vector<std::vector<complex<float>>> globalFDVector;
    std::vector<complex<float>> temp_vector;
    ifstream DBFDData ("/home/isadora/Desktop/finaltest/DBimgs/DBFDData.txt");
    std::complex<float> c;

    if (DBFDData.is_open())
    {
      while ( getline (DBFDData,line) )
      {
          std::stringstream stream(line);
                  while(stream>>c)
                  {
                     vec.push_back(c);
                  }
      }
      myfile.close();
    }

}

faceType getCategory(int index){

    if(index == IM_PER_CLASS*mul) mul++;

    return faceType(mul-1);
}

void registerFD(std::vector<complex<float>> imDBFD, faceType fType, ofstream& DBFDData){

    for (int i = 0 ; i < imDBFD.size(); i++){
        if(i == 0){
            DBFDData << fType <<"\n";
            DBFDData << imDBFD.at(i)<<"\n";
        }else{
        DBFDData << imDBFD.at(i)<<"\n";
        }
    }
}

void extractAverageColour(std::vector<Point2d> contour, Mat inputIM){
    // Define a polygon
    Point pts[1][contour.size()];
    for(unsigned int i = 0; i < contour.size(); i++) pts[0][i] = Point (contour.at(i).x,contour.at(i).y);

    const Point* points[1] = {pts[0]};
    int npoints = contour.size();

    // Create the mask with the polygon
    Mat1b mask(inputIM.rows, inputIM.cols, uchar(0));
    fillPoly(mask, points, &npoints, 1, Scalar(255));

    // Compute the mean with the computed mask (BGR)
    Scalar average = mean(inputIM, mask);


    Mat bla;
    inputIM.copyTo(bla, mask);
    imwrite( "/home/isadora/Desktop/finaltest/DBimgs/Gray_Image.jpg", bla );
    std::cout << "Average colour (RGB): "<< average[2] <<" "<< average[1] <<" "<< average[0] << std::endl;
}

double compareFD(std::vector<complex<float>> fd1, std::vector<complex<float>> fd2){
    double temp_diff = 0.0f;

    if (fd1.size() != fd2.size())
        return -1;

    for(unsigned int i = 0; i < fd1.size(); i++){
        temp_diff += pow(abs(fd1.at(i).real() - fd2.at(i).real()),2);
    }
    return sqrt(temp_diff);
}


std::vector<complex<float>> getFD(std::vector<Point2d> contour, Mat silImage){
    int n_pixels_contour = contour.size();

    ///find the centroid
    Moments m = moments(silImage, true);
    Point center(m.m10/m.m00, m.m01/m.m00);

    ///create centroid distance function
    std::vector<float> cdf;
    for(int i = 0; i < n_pixels_contour; i++)
        cdf.push_back( sqrt(pow(contour[i].x-center.x,2) + pow(contour[i].y-center.y,2)) );

    ///create FDS coefficients
    complex<float> j;
    j = -1;
    j = sqrt(j);
    complex<float> result;
    std::vector<complex<float>> fd, fourier_coeff;
    float norm_factor;


    for (int n = 0; n < n_pixels_contour; ++n)
    {
        result = 0.0f;
        for (int t = 0; t < n_pixels_contour; ++t)
        {
            result += (cdf[t] * exp(-j*((float)M_PI)*((float)n)*((float)t) / ((float)n_pixels_contour)));

        }

        fourier_coeff.push_back((1.0f / n_pixels_contour) * result);
    }


    fourier_coeff.at(0) = complex<float>(0,0);

    norm_factor = abs(fourier_coeff.at(1));

    for( unsigned int i = 0; i < fourier_coeff.size(); i++) {
        fd.push_back(abs(fourier_coeff.at(i))/norm_factor);
    }

    return fd;
}

std::vector<Point2d> getContour(Mat gray){


    std::vector<std::vector<Point>> contours;
    std::vector<Point2d> filt_contours;

    findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    int biggest = 0;
    for (unsigned int i = 1; i < contours.size(); i++){
        if(contours[i].size()>contours[biggest].size())
            biggest = i;
    }

    ///shape size normalization
    int step = (int)(ceil(contours[0].size()/candidate_points));
    for (unsigned int i = 0, j = 0; i < contours[biggest].size() && j < candidate_points; i += step, j++){
        //filt_contours.push_back(contours[biggest][i]);
        filt_contours.push_back(Point2d(contours[biggest][i].x, contours[biggest][i].y));
    }

    return filt_contours;
}

Mat getSilhouette(frontal_face_detector detector, shape_predictor sp, string fileName, std::vector<Point2d>& landmarks){
    array2d<rgb_pixel> image;

    load_image(image, fileName);
    std::vector<dlib::rectangle> dets = detector(image);
    Point contour[1][27];
        full_object_detection face;
    std::vector<Point2d> contourVector, filt_contours;
        int cd = LANDS-1;
        std::vector<Point2d> contourFD;

    std::vector<full_object_detection> shapes;

    for (unsigned long j = 0; j < dets.size(); ++j)
     {
         full_object_detection shape = sp(image, dets[j]);
         shapes.push_back(shape);
         face = shape;

     }

    for(int k = 0; k < LANDS; k++){
        if (k <= 16) contour[0][k] = Point(face.part(k).x(),face.part(k).y());
        else{
            contour[0][cd] = Point(face.part(k).x(),face.part(k).y());
            cd--;
        }
    }

    for(int k = 0; k <68 ; k++) landmarks.push_back(Point2d(face.part(k).x(),face.part(k).y()));
    for(int l = 0; l <LANDS ; l++) contourFD.push_back(Point2d (contour[0][l].x,contour[0][l].y));


    Mat imageCV;
    imageCV = imread(fileName);
    const Point* ppt[1] = { contour[0] };

    int npt[] = { LANDS };
    fillPoly( imageCV, ppt, npt, 1, Scalar( 0, 0, 0 ), 1 );

    Mat mask(imageCV.rows, imageCV.cols, CV_8UC3, Scalar(255,255,255));
    fillPoly( mask, ppt, npt, 1, Scalar( 0, 0, 0 ), 1 );

    Mat gray;
    cvtColor(mask, gray, CV_BGR2GRAY);
    threshold(gray, gray, 1, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    bitwise_not (gray, gray);

    //WE HAVE THE SILHOUETTE SAVED IN <MASK>
    return gray;
}


Mat getSilhouetteDB(frontal_face_detector detector, shape_predictor sp, string fileName){
    array2d<rgb_pixel> image;

    load_image(image, fileName);
    std::vector<dlib::rectangle> dets = detector(image);
    Point contour[1][27];
        full_object_detection face;
    std::vector<Point2d> contourVector, filt_contours;
        int cd = LANDS-1;
        std::vector<Point2d> contourFD;

    std::vector<full_object_detection> shapes;

    for (unsigned long j = 0; j < dets.size(); ++j)
     {
         full_object_detection shape = sp(image, dets[j]);
         shapes.push_back(shape);
         face = shape;

     }

    for(int k = 0; k < LANDS; k++){
        if (k <= 16) contour[0][k] = Point(face.part(k).x(),face.part(k).y());
        else{
            contour[0][cd] = Point(face.part(k).x(),face.part(k).y());
            cd--;
        }
    }

    for(int l = 0; l <LANDS ; l++) contourFD.push_back(Point2d (contour[0][l].x,contour[0][l].y));


    Mat imageCV;
    imageCV = imread(fileName);
    const Point* ppt[1] = { contour[0] };

    int npt[] = { LANDS };
    fillPoly( imageCV, ppt, npt, 1, Scalar( 0, 0, 0 ), 1 );

    Mat mask(imageCV.rows, imageCV.cols, CV_8UC3, Scalar(255,255,255));
    fillPoly( mask, ppt, npt, 1, Scalar( 0, 0, 0 ), 1 );

    Mat gray;
    cvtColor(mask, gray, CV_BGR2GRAY);
    threshold(gray, gray, 1, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    bitwise_not (gray, gray);

    //WE HAVE THE SILHOUETTE SAVED IN <MASK>
    return gray;
}
