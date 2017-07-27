#include "cmshapecontext.h"

const double CmShapeContext::SC_INNER_R = 0.125; // Regions = 0.125 * (1,2,4,8,16)
const double CmShapeContext::DUMMY_COST = 0.25;	 // Dummy matching cost
bool CmShapeContext::bDisplay = false;
int CmShapeContext::nIters = 1;
double CmShapeContext::dummyFrac = 0.25;
bool CmShapeContext::findCdMap = false;

void CmShapeContext::Config(bool _display, int _iters, double _dummyFraction)
{
    bDisplay = _display;
    nIters = _iters;
    dummyFrac = _dummyFraction;
}

double CmShapeContext::matchCost(PointSetd &pntSet1,PointSetd &pntSet2, double wsc)
{
    double affcost, scCost;
    shapeMatch(pntSet1, pntSet2, scCost, affcost);
    double scCost1 = affcost * (1-wsc) + scCost * wsc;
    for (size_t i = 0; i < pntSet1.size(); i++)
        pntSet1[i].x = -pntSet1[i].x;
    shapeMatch(pntSet1, pntSet2, scCost, affcost);
    double scCost2 = affcost * (1-wsc) + scCost * wsc;
    return min(scCost1, scCost2);
}

double CmShapeContext::shapeMatch(PointSetd &pntSet1, PointSetd &pntSet2, double &scCost, double &affCost)
{
    CV_Assert(pntSet1.size() == S_NU_SC && pntSet2.size() == S_NU_SC);

    int	n = pntSet1.size();
    int	m = pntSet2.size();
    int	leng = cvRound(max(n, m) * (1 + dummyFrac));
    int	i, j;

    vector<ShapeContext> scData1(n);
    vector<ShapeContext> scData2(m);

    bool *alive1, *alive2, *mx, *my;
    int	*cx, *cy;
    double	*lx, *ly;
    double	**cost;

    alive1	= new bool[leng];
    alive2	= new bool[leng];
    mx	= new bool[leng];
    my	= new bool[leng];
    cx	= new int[leng];
    cy	= new int[leng];
    lx	= new double[leng];
    ly	= new double[leng];

    cost = new double*[leng];
    for (i = 0; i < leng; i ++)
        cost[i] = new double[leng];

    memset(alive1, 1, leng);
    memset(alive2, 1, leng);

    PointSetd	curtPntSet = pntSet1;

    vecD	scCostList;
    vecD	affCostList;

    PointSetd	vpA, vpB;
    double scale2 = 1.0, scale1 = 1.0;

    for (int it = 1; it <= nIters; it ++)
    {
        double	meanDist1 = calcShapeContextData(curtPntSet, scData1, alive1);
        double	meanDist2 = calcShapeContextData(pntSet2, scData2, alive2);
        scale1 *= meanDist1;
        scale2 *= meanDist2;
        if (gridPnts.size())
        {
            for (size_t i = 0; i < gridPnts.size(); i++)
            {
                gridPnts[i].x /= meanDist1;
                gridPnts[i].y /= meanDist1;
            }
        }

        for (i = 0; i < leng; i ++)
            for (j = 0; j < leng; j ++)
                if (i >= n || j >= m)
                    cost[i][j] = DUMMY_COST;
                else
                    cost[i][j] = shapeContextCost(scData1[i], scData2[j]);

        double	matchCost = hungaryMatch(cost, cx, cy, lx, ly, mx, my, leng);

        if (bDisplay)
            displayMatchingResult(curtPntSet, pntSet2, cx);

        double	rowLower, rowMin;
        double	colLower, colMin;

        rowLower = 0;
        for (i = 0; i < n; i ++)
        {
            rowMin = INF;
            for (j = 0; j < m; j ++)
                if (cost[i][j] < rowMin) rowMin = cost[i][j];
            rowLower += rowMin;
        }
        colLower = 0;
        for (j = 0; j < m; j ++)
        {
            colMin = INF;
            for (i = 0; i < n; i ++)
                if (cost[i][j] < colMin) colMin = cost[i][j];
            colLower += colMin;
        }
        rowLower /= n; colLower /= m;
        scCost = max(rowLower, colLower);
        scCostList.push_back( scCost );

        vpA.clear();
        vpB.clear();
        for (i = 0; i < n; i ++)
        {
            if (cx[i] < m)
            {
                vpA.push_back( curtPntSet[i] );
                vpB.push_back( pntSet2[ cx[i] ] );
            }
        }

        double	beta_k = sqr(meanDist2) * (it == 1 ? 1000 : 1);
        findCdMap = (it == nIters);
        affCost = calcAffineTransformation(curtPntSet, pntSet2, vpA, vpB, beta_k);
        affCostList.push_back(affCost);

        for (i = 0; i < n; i ++)
            alive1[i] = (cx[i] < m);
        for (i = 0; i < m; i ++)
            alive2[i] = (cy[i] < n);
    }

    delete[] alive1;
    delete[] alive2;
    delete[] mx;
    delete[] my;
    delete[] cx;
    delete[] cy;
    delete[] lx;
    delete[] ly;

    for (i = 0; i < leng; i ++)
        delete[] cost[i];
    delete[] cost;

    if (gridPnts.size())
    {
        for (size_t i = 0; i < gridPnts.size(); i++)
        {
            gridPnts[i].x *= scale2;
            gridPnts[i].y *= scale2;
        }
    }

    return scale1 / scale2;
}

/*************** Logging & Display *****************/

void CmShapeContext::displayMatchingResult(const PointSetd& pntSet1, const PointSetd& pntSet2, int *match)
{
    //PointSeti pts1, pts2;
    //int dSize = 600;
    //CmShow::Normalize(pntSet1, pts1, dSize);
    //CmShow::Normalize(pntSet2, pts2, dSize);

    //Mat showImg = Mat::zeros(dSize, dSize, CV_8UC3);
    //int n1 = (int)pts1.size(), n2 = (int)pts2.size();
    //for (int i = 0; i < n2; i++)
    //	circle(showImg, pts2[i], 2, CmShow::RED);

    //for (int i = 0; i < n1; i++)
    //{
    //	circle(showImg, pts1[i], 2, CmShow::GREEN);
    //	if (match[i] < n2)
    //		line(showImg, pts1[i], pts2[match[i]], CmShow::YELLOW);
    //}
    //imshow("Match result", showImg);
    //waitKey(0);
}

void CmShapeContext::displayAffine(const PointSetd& grid, const PointSetd& pntSet1, const PointSetd& pntSet2)
{
    //Mat showImg;
    //CmShow::PntSet(grid, CmShow::GRAY, "", showImg);
    //CmShow::PntSet(pntSet1, CmShow::GREEN, "", showImg);
    //CmShow::PntSet(pntSet2, CmShow::RED, "Display Affine", showImg);
    //waitKey(0);
}

double CmShapeContext::calcShapeContextData(PointSetd& pntSet, vector<ShapeContext>& scData, bool *alive)
{
    int	n = pntSet.size();
    int	i, j, cc = 0;
    double	meanDist = 0;

    for (i = 0; i < n; i ++)
        if (alive[i])
            for (j = i + 1; j < n; j ++)
                if (alive[j])
                    meanDist += pntDist(pntSet[i], pntSet[j]), cc ++;
    meanDist /= cc;

    // ¾àÀë¹æ·¶»¯
    for (i = 0; i < n; i ++)
    {
        pntSet[i].x /= meanDist;
        pntSet[i].y /= meanDist;
    }

    double	d, lv, theta;
    double	sectorAngle = CV_PI * 2.0 / SC_SECTOR;
    int	r, s;

    // ¼ÆËãShape Context
    for (i = 0; i < n; i ++)
    {
        memset(scData[i].h, 0, sizeof(scData[i].h));
        for (j = 0; j < n; j ++)
            if (alive[j] && j != i)
            {
                d = pntDist(pntSet[i], pntSet[j]);
                for (r = 0, lv = SC_INNER_R; r < SC_REGION; r ++, lv += lv)
                    if (d <= lv) break;
                if (r < SC_REGION)
                {
                    theta = atan2(pntSet[j].y - pntSet[i].y, pntSet[j].x - pntSet[i].x);
                    for (s = 0, lv = - CV_PI + sectorAngle; s + 1 < SC_SECTOR; s ++, lv += sectorAngle)
                        if (theta <= lv) break;
                    scData[i].h[r][s] ++;
                }
            }
    }

    return	meanDist;
}

double CmShapeContext::shapeContextCost(const ShapeContext& A, const ShapeContext& B)
{
    double	ret = 0;
    double	ca = EPS, cb = EPS;

    for (int r = 0; r < SC_REGION; r ++)
        for (int s = 0; s < SC_SECTOR; s ++)
            ca += A.h[r][s], cb += B.h[r][s];

    // Both no neighbors
    if (ca < 1 - EPS && cb < 1 - EPS) return 0;

    for (int r = 0; r < SC_REGION; r ++)
        for (int s = 0; s < SC_SECTOR; s ++)
            if (A.h[r][s] || B.h[r][s])
                ret += sqr(A.h[r][s] / ca - B.h[r][s] / cb) / (A.h[r][s] / ca + B.h[r][s] / cb);
    return ret / 2.0;
}

bool CmShapeContext::hungaryExtendPath(int u, double **cost, int *cx, int *cy, double* lx, double* ly, bool *mx, bool *my, int n)
{
    mx[u] = 1;
    for (int v = 0; v < n; v ++)
        if (CmSgn(cost[u][v] - (lx[u] + ly[v])) == 0 && ! my[v])
        {
            my[v] = 1;
            if (cy[v] < 0 || hungaryExtendPath(cy[v], cost, cx, cy, lx, ly, mx, my, n))
                return cx[u] = v, cy[v] = u, true;
        }
    return false;
}

double CmShapeContext::hungaryMatch(double **cost, int *cx, int *cy, double* lx, double* ly, bool *mx, bool *my, int n)
{
    int	u, i, j;
    double	alpha;

    for (i = 0; i < n; i ++)
    {
        cx[i] = cy[i] = -1;
        lx[i] = INF;
        ly[i] = 0;

        for (j = 0; j < n; j ++)
            if (cost[i][j] < lx[i])
                lx[i] = cost[i][j];
    }

    for (u = 0; u < n; u ++)
    {
        while (cx[u] < 0)
        {
            memset(mx, 0, n);
            memset(my, 0, n);

            if (hungaryExtendPath(u, cost, cx, cy, lx, ly, mx, my, n))
                break;

            alpha = INF;
            for (i = 0; i < n; i ++)
                if (mx[i])
                    for (j = 0; j < n; j ++)
                        if (!my[j])
                            alpha = min(alpha, cost[i][j] - lx[i] - ly[j]);
            for (i = 0; i < n; i ++)
                if (mx[i]) lx[i] += alpha;
            for (i = 0; i < n; i ++)
                if (my[i]) ly[i] -= alpha;
        }
    }

    double	matchCost = 0;
    for (i = 0; i < n; i ++)
        matchCost += lx[i] + ly[i];

    return	matchCost;
}

double CmShapeContext::calcAffineTransformation(PointSetd& curtPntSet, const PointSetd& tarPntSet, const PointSetd& pA, const PointSetd& pB, double beta_k)
{
    int	n = pA.size();
    int	i, j;

    CvMat	*L = cvCreateMat(n + 3, n + 3, CV_32FC1);
    CvMat	*invL = cvCreateMat(n + 3, n + 3, CV_32FC1);
    CvMat	*V = cvCreateMat(n + 3, 2, CV_32FC1);

    // Set L
    for (i = 0; i < n; i ++)
    {
        for (j = 0; j < n; j ++)
        {
            if (i == j)
            {
                cvmSet(L, i, j, beta_k);
            }
            else
            {
                double	d = pntSqrDist(pA[i], pA[j]);
                cvmSet(L, i, j, d * log(d));
            }
        }
    }

    for (i = 0; i < n; i ++)
    {
        cvmSet(L, i, n, 1.0);
        cvmSet(L, i, n+1, pA[i].x);
        cvmSet(L, i, n+2, pA[i].y);

        cvmSet(L, n, i, 1.0);
        cvmSet(L, n+1, i, pA[i].x);
        cvmSet(L, n+2, i, pA[i].y);
    }

    for (i = n; i < n + 3; i ++)
        for (j = n; j < n + 3; j ++)
            cvmSet(L, i, j, 0);

    // Set V
    for (i = 0; i < n; i ++)
    {
        cvmSet(V, i, 0, pB[i].x);
        cvmSet(V, i, 1, pB[i].y);
    }
    for (i = n; i < n + 3; i ++)
    {
        cvmSet(V, i, 0, 0);
        cvmSet(V, i, 1, 0);
    }

    cvInvert(L, invL);

    CvMat *C = cvCreateMat(n+3, 2, CV_32FC1);
    cvMatMul(invL, V, C);

    // bennding energy
    double	E = 0;
    for (int p = 0; p < 2; p ++)
    {
        double	val = 0;
        for (i = 0; i < n; i ++)
            for (j = 0; j < n; j ++)
                if (i != j)
                    val += cvmGet(C, i, p) * cvmGet(L, i, j) * cvmGet(C, j, p);
        E += val;
    }
    E /= 2.0;

    CvMat	*A = cvCreateMat(2, 2, CV_32FC1);
    cvmSet(A, 0, 0, cvmGet(C, n + 1 , 0));
    cvmSet(A, 1, 0, cvmGet(C, n + 2 , 0));
    cvmSet(A, 0, 1, cvmGet(C, n + 1 , 1));
    cvmSet(A, 1, 1, cvmGet(C, n + 2 , 1));

    CvMat	*S = cvCreateMat(2, 1, CV_32FC1);
    cvSVD(A, S);

    double affCost = log((cvmGet(S, 0, 0) + 1e-20) / (cvmGet(S, 1, 0) + 1e-20));

    double	* cx = new double[n + 3];
    double	* cy = new double[n + 3];

    for (i = 0; i < n + 3; i ++)
    {
        cx[i] = cvmGet(C, i, 0);
        cy[i] = cvmGet(C, i, 1);
    }

    //// DO Affine Transformation & Display

    makeAffineTransformation(curtPntSet, pA, cx, cy);

    if (bDisplay && gridPnts.size() == 0)
    {
        gridPnts.resize(30 * 30);
        int		cc = 0;

        double		size = 0;
        for (i = 0; i < (int)curtPntSet.size(); i ++)
            size = max(size, max(curtPntSet[i].x, curtPntSet[i].y));
        for (i = 0; i < (int)tarPntSet.size(); i ++)
            size = max(size, max(tarPntSet[i].x, tarPntSet[i].y));
        size /= 25;
        for (i = 0; i < 30; i ++)
            for (j = 0; j < 30; j ++)
            {
                gridPnts[cc].x = i * size;
                gridPnts[cc].y = j * size;
                cc ++;
            }

    }

    if (gridPnts.size() > 0 && (bDisplay || findCdMap))
        makeAffineTransformation(gridPnts, pA, cx, cy);

    if (bDisplay)
        displayAffine(gridPnts, curtPntSet, tarPntSet);

    delete[] cx;
    delete[] cy;

    cvReleaseMat(&L);
    cvReleaseMat(&invL);
    cvReleaseMat(&V);
    cvReleaseMat(&C);
    cvReleaseMat(&A);
    cvReleaseMat(&S);
    return affCost;
}

void CmShapeContext::makeAffineTransformation(PointSetd& pntSet, const PointSetd& pA, const double *cx, const double *cy)
{
    int	leng = max(pntSet.size(), pA.size() + 3);
    double	* aff = new double[ leng ];
    double	* wrp = new double[ leng ];

    PointSetd	dst(pntSet.size());

    int	i , j;
    int	n = pntSet.size();
    int	n_good = pA.size();

    double	d2, u;

    for (i = 0; i < n; i ++)
        aff[i] = cx[n_good] + cx[n_good + 1] * pntSet[i].x + cx[n_good + 2] * pntSet[i].y;

    memset(wrp, 0, leng * sizeof(double));
    for (i = 0; i < n_good; i ++)
        for (j = 0; j < n; j ++)
        {
            d2 = pntSqrDist(pA[i], pntSet[j]);
            u = d2 * log(d2 + EPS);
            wrp[j] += cx[i] * u;
        };
    for (i = 0; i < n; i ++)
        dst[i].x = aff[i] + wrp[i];

    for (i = 0; i < n; i ++)
        aff[i] = cy[n_good] + cy[n_good + 1] * pntSet[i].x + cy[n_good + 2] * pntSet[i].y;

    memset(wrp, 0, leng * sizeof(double));
    for (i = 0; i < n_good; i ++)
        for (j = 0; j < n; j ++)
        {
            d2 = pntSqrDist(pA[i], pntSet[j]);
            u = d2 * log(d2 + EPS);
            wrp[j] += cy[i] * u;
        };
    for (i = 0; i < n; i ++)
        dst[i].y = aff[i] + wrp[i];

    pntSet = dst;
    delete[] aff;
    delete[] wrp;
}




