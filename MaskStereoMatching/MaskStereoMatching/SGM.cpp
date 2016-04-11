
#include "SGM.h"
#include <highgui.h>
using namespace cv;
using namespace std;
void SGM::rawCostCalculate()
{
	int H = m_imgL.rows;
	int W = m_imgL.cols;

	if (m_rawCostCube != NULL) delete m_rawCostCube;
	
	int size[3];
	size[0] = m_dispLevels; size[1] = H; size[2] = W;
	double val = 1e5;
	m_rawCostCube = new cv::Mat(3,size,CV_64FC1,Scalar(val));
	/*imshow("tst",m_imgL);
	waitKey(0);*/
	for(int d=0; d<m_dispLevels; d++)
	{
		for(int i=0; i<H; i++)
		{
			for(int j=0; j<W; j++)
			{
				int jr = max(0,j-d);
				m_rawCostCube->at<double>(d,i,j) = abs(( double)m_imgL.at<uchar>(i,j) - ( double)m_imgR.at<uchar>(i,jr) );
				
			}
		}
	}	
		
}

void SGM::pathEvaluate(int x,int y, int x_r, int y_r)//void SGM::pathEvaluate(int x,int y,int d, int x_r, int y_r)
{
	for(int d=0; d<m_dispLevels; d++)
	{
		double smooth_term = 1e20;// 
		for(int d_p=0; d_p<m_dispLevels; d_p++)
		{
			double priorCost = m_sgmCostCube->at<double>(d_p,y_r,x_r);//double priorCost = m_sgmCostCube->at<double>(d,y_r,x_r);

			if (d_p == d)
			{
				smooth_term = min(smooth_term,priorCost);
			}
			else if (abs(d_p - d) == 1)
			{
				smooth_term = min(smooth_term, priorCost + m_P1);
			}
			else 
			{
				double path_intensity_grad = abs( ( double)m_imgL.at<uchar>(y,x) - (double)m_imgL.at<uchar>(y_r,x_r));
				double pp =0;
				if (path_intensity_grad >0.001) pp = m_P2/path_intensity_grad;
				else pp = m_P2;

				smooth_term = min(smooth_term, priorCost +
								max(m_P1, pp)
								);
			}
		}
		m_sgmCostCube->at<double>(d,y,x) += m_rawCostCube->at<double>(d,y,x) + smooth_term;//m_sgmCostCube->at<double>(d,y,x) = m_rawCostCube->at<double>(d,y,x) + smooth_term;

	}

	int m = 1e20;
	for(int d_p=0; d_p<m_dispLevels; d_p++)
	{
		if (m_sgmCostCube->at<double>(d_p,y_r,x_r) < m) m = m_sgmCostCube->at<double>(d_p,y_r,x_r);
	}
	assert(m != 1e20);
	for(int dd=0; dd<m_dispLevels; dd++)
		m_sgmCostCube->at<double>(dd,y,x) -= m ;

}
void SGM::sgmCostCalculate()
{
	int H = m_imgL.rows;
	int W = m_imgL.cols;

	if(m_sgmCostCube != NULL) delete m_sgmCostCube;

	int size[3];
	size[0] = m_dispLevels; size[1] = H; size[2] = W;
	double val = 0;//1e5;//
	m_sgmCostCube = new Mat(3,size,CV_64FC1,Scalar(val));

	//从上到下
	for(int y=0; y<H; y++)
	{
		for(int x=0; x<W; x++)
		{
			for(int d=0; d<m_dispLevels; d++)
			{		
				int x_r=0,y_r=0;
				//if ()
				//left
				//if(x>=1)
				{
					x_r = max(x-1,0); y_r = y;
					pathEvaluate(x, y, x_r, y_r);
				}
				
				//top-left
				//if (x>=1 && y>=1)
				{
					x_r = max(x-1,0); y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r);
				}
				
				//top
				//if (y>=1)
				{
					x_r = x; y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r);
				}
				
				//top-right
			//	if (y>=1 && x<W-1)
				{
					x_r = min(x+1,W-1); y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r);
				}
				
			}
		}
	}

	//从下到上
	for(int y=H-1; y>=0; y--)
	{
		for(int x=W-1; x>=0; x--)
		{
			for(int d=0; d<m_dispLevels; d++)
			{		
				int x_r=0,y_r=0;
				//if ()
				//right
				x_r = min(x+1,W-1); y_r = y;
				pathEvaluate(x, y,  x_r, y_r);
				//bottom-right
				x_r = min(x+1,W-1); y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r);
				//bottom
				x_r = x; y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r);
				//bottom-left
				x_r = max(x-1,0); y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r);
			}
		}
	}
	

	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			double min = 1e20;
			int d_min = 0;
			for(int d=0; d<m_dispLevels; d++)
			{
				double cost_t = m_sgmCostCube->at<double>(d,i,j);
				if (cost_t < min)
				{
					min = cost_t;
					d_min = d;
				}
			}
			m_dispImg.at<double>(i,j) = d_min;
		}
	}

}

void SGM::sgmRun()
{
	rawCostCalculate();
	sgmCostCalculate();
}