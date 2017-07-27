#ifndef CMSHAPECONTEXT_H
#define CMSHAPECONTEXT_H


const int S_NU_SC = 64; //Sample number of shape context

#include <vector>
#include <opencv/cv.h>
using namespace cv;
using namespace std;


typedef vector<Point2d> PointSetd;
typedef vector<double> vecD;
const double EPS = 1e-10;		// Epsilon (zero value)
const double INF = 1e200;
template<typename T> inline int CmSgn(T number) {if(abs(number) < EPS) return 0; return number > 0 ? 1 : -1; }
template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T> inline T pntSqrDist(const Point_<T> &p1, const Point_<T> &p2) {return sqr(p1.x - p2.x) + sqr(p1.y - p2.y);} // out of range risk for T = byte, ...
template<class T> inline double pntDist(const Point_<T> &p1, const Point_<T> &p2) {return sqrt((double)pntSqrDist(p1, p2));} // out of range risk for T = byte, ...


class	CmShapeContext
{
    static const int SC_SECTOR = 12;
    static const int SC_REGION = 5;	// 5 * 12 bins(as ShapeContext Definition)
    static const double SC_INNER_R;	// Regions = 0.125 * (1,2,4,8,16)
    static const double DUMMY_COST;	// Dummy matching cost

    struct	ShapeContext { char h[SC_REGION][SC_SECTOR]; };

    /* configuration variables */
    static bool bDisplay; // whether display figures
    static int nIters;   // number of iterations
    static double dummyFrac; // fraction of dummy points
    static bool findCdMap;

public:
    /*************** Configuration methods *****************/
    static void Config(bool display = false, int iters = 1, double dummyFraction = 0.25);

    PointSetd gridPnts;


    /*************** Service methods *****************/
    double shapeMatch(PointSetd &pntSet1,PointSetd &pntSet2, double& scCost, double& affCost);

    double matchCost(PointSetd &pntSet1,PointSetd &pntSet2, double wsc = 0.95);

private:

    /*************** Logging & Display *****************/
    static void displayMatchingResult(const PointSetd& pntSet1, const PointSetd& pntSet2, int *match);
    static void displayAffine(const PointSetd& grid, const PointSetd& pntSet1, const PointSetd& pntSet2);


    /*************** Private Methods *****************/
    static double calcShapeContextData(PointSetd& pntSet, vector<ShapeContext>& scData, bool *alive);
    static double shapeContextCost(const ShapeContext& A, const ShapeContext& B);
    static double hungaryMatch(double **cost, int *cx, int *cy, double* lx, double* ly, bool *mx, bool *my, int n);
    static bool hungaryExtendPath(int u, double **cost, int *cx, int *cy, double* lx, double* ly, bool *mx, bool *my, int n);
    double calcAffineTransformation(PointSetd& curtPntSet, const PointSetd& tarPntSet, const PointSetd& pA, const PointSetd& pB, double beta_k);
    static void makeAffineTransformation(PointSetd& pntSet, const PointSetd& pA, const double *cx, const double *cy);
};

#endif // CMSHAPECONTEXT_H
