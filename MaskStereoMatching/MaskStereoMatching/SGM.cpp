
#include "SGM.h"
#include <highgui.h>
#include "time.h"
#include <iostream>
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
	for(int di=0; di<m_dispLevels; di++)//for(int d=0; d<m_dispLevels; d++)
	{
		int d = di + m_dispMin;
		for(int i=0; i<H; i++)
		{
			for(int j=0; j<W; j++)
			{
				//if (useMask && m_maskImg.at<uchar>(i,j) == 0) continue;
				int jr = max(0,j-d);
	
				m_rawCostCube->at<double>(di,i,j) = abs(( double)m_imgL.at<uchar>(i,j) - ( double)m_imgR.at<uchar>(i,jr) );//m_rawCostCube->at<double>(d,i,j) = abs(( double)m_imgL.at<uchar>(i,j) - ( double)m_imgR.at<uchar>(i,jr) );
				
			}
		}
	}	
		
}
void SGM::pathEvaluate(int x,int y, int x_r, int y_r,cv::Mat * Lr)
{
	for(int di=0; di<m_dispLevels; di++)
	{
		int d = di + m_dispMin;
		double smooth_term = 1e20;// 
		for(int d_pi=0; d_pi<m_dispLevels; d_pi++)//for(int d_p=0; d_p<m_dispLevels; d_p++)
		{
			int d_p = d_pi + m_dispMin;
			double priorCost = Lr->at<double>(d_pi,y_r,x_r);//double priorCost = Lr->at<double>(d_p,y_r,x_r);//double priorCost = m_sgmCostCube->at<double>(d,y_r,x_r);

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
		Lr->at<double>(di,y,x) += m_rawCostCube->at<double>(di,y,x) + smooth_term;//Lr->at<double>(d,y,x) += m_rawCostCube->at<double>(d,y,x) + smooth_term;//m_sgmCostCube->at<double>(d,y,x) = m_rawCostCube->at<double>(d,y,x) + smooth_term;

	}


	int m = 1e20;
	for(int d_pi=0; d_pi<m_dispLevels; d_pi++)
	{
		if (Lr->at<double>(d_pi,y_r,x_r) < m) m = Lr->at<double>(d_pi,y_r,x_r);
	}
	assert(m != 1e20);
	for(int dd=0; dd<m_dispLevels; dd++)
	{
		Lr->at<double>(dd,y,x) -= m ;
	}

}
//void SGM::pathEvaluate(int x,int y, int x_r, int y_r)//void SGM::pathEvaluate(int x,int y,int d, int x_r, int y_r)
//{
////clock_t timer;static int cnt=0;if (cnt++%10000==0)timer = clock();
//
//	for(int d=0; d<m_dispLevels; d++)
//	{
//		double smooth_term = 1e20;// 
//		for(int d_p=0; d_p<m_dispLevels; d_p++)
//		{
//			double priorCost = m_sgmCostCube->at<double>(d_p,y_r,x_r);//double priorCost = m_sgmCostCube->at<double>(d,y_r,x_r);
//
//			if (d_p == d)
//			{
//				smooth_term = min(smooth_term,priorCost);
//			}
//			else if (abs(d_p - d) == 1)
//			{
//				smooth_term = min(smooth_term, priorCost + m_P1);
//			}
//			else 
//			{
//				double path_intensity_grad = abs( ( double)m_imgL.at<uchar>(y,x) - (double)m_imgL.at<uchar>(y_r,x_r));
//				double pp =0;
//				if (path_intensity_grad >0.001) pp = m_P2/path_intensity_grad;
//				else pp = m_P2;
//
//				smooth_term = min(smooth_term, priorCost +
//								max(m_P1, pp)
//								);
//			}
//		}
//		m_sgmCostCube->at<double>(d,y,x) += m_rawCostCube->at<double>(d,y,x) + smooth_term;//m_sgmCostCube->at<double>(d,y,x) = m_rawCostCube->at<double>(d,y,x) + smooth_term;
//
//	}
//
//
//	int m = 1e20;
//	for(int d_p=0; d_p<m_dispLevels; d_p++)
//	{
//		if (m_sgmCostCube->at<double>(d_p,y_r,x_r) < m) m = m_sgmCostCube->at<double>(d_p,y_r,x_r);
//	}
//	assert(m != 1e20);
//	for(int dd=0; dd<m_dispLevels; dd++)
//	{
//		m_sgmCostCube->at<double>(dd,y,x) -= m ;
//	}
//
// //if (cnt%100000 ==100000-1) cout<<clock()-timer<<endl;
//}
void SGM::sgmCostCalculate()
{
	int H = m_imgL.rows;
	int W = m_imgL.cols;

	if(m_sgmCostCube != NULL) delete m_sgmCostCube;

	int size[3];
	size[0] = m_dispLevels; size[1] = H; size[2] = W;
	double val = 1e6;//1e5;//
	m_sgmCostCube = new Mat(3,size,CV_64FC1,Scalar(0));
	m_Lr0 =  new Mat(3,size,CV_64FC1,Scalar(val));
	m_Lr1 =  new Mat(3,size,CV_64FC1,Scalar(val));
	m_Lr2 =  new Mat(3,size,CV_64FC1,Scalar(val));
	m_Lr3 =  new Mat(3,size,CV_64FC1,Scalar(val));

	
	//从上到下
	for(int y=0; y<H; y++)
	{
		for(int x=0; x<W; x++)
		{
			//if (useMask && m_maskImg.at<uchar>(y,x) == 0) continue;
			//for(int d=0; d<m_dispLevels; d++)
			//{		
				int x_r=0,y_r=0;
				//if ()
				//left
				//if(x>=1)
				{
					x_r = max(x-1,0); y_r = y;
					pathEvaluate(x, y, x_r, y_r,m_Lr0);
				}
				
				//top-left
				//if (x>=1 && y>=1)
				{
					x_r = max(x-1,0); y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r,m_Lr1);
				}
				
				//top
				//if (y>=1)
				{
					x_r = x; y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r,m_Lr2);
				}
				
				//top-right
			//	if (y>=1 && x<W-1)
				{
					x_r = min(x+1,W-1); y_r = max(y-1,0);
					pathEvaluate(x, y, x_r, y_r,m_Lr3);
				}
				
			//}
		}
	}
	for(int i=0;i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			for(int di=0;di<m_dispLevels; di++)
			{
				m_sgmCostCube->at<double>(di,i,j) +=  m_Lr0->at<double>(di,i,j)
					+ m_Lr1->at<double>(di,i,j)
					+ m_Lr2->at<double>(di,i,j)
					+ m_Lr3->at<double>(di,i,j);
			}
		}
	}
	//从下到上
	for(int y=H-1; y>=0; y--)
	{
		for(int x=W-1; x>=0; x--)
		{
			//if (useMask && m_maskImg.at<uchar>(y,x) == 0) continue;
			//for(int d=0; d<m_dispLevels; d++)
			//{		
				int x_r=0,y_r=0;
				//if ()
				//right
				x_r = min(x+1,W-1); y_r = y;
				pathEvaluate(x, y,  x_r, y_r,m_Lr0);
				//bottom-right
				x_r = min(x+1,W-1); y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r,m_Lr1);
				//bottom
				x_r = x; y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r,m_Lr2);
				//bottom-left
				x_r = max(x-1,0); y_r = min(y+1,H-1);
				pathEvaluate(x, y, x_r, y_r,m_Lr3);
			//}
		}
	}
	
	for(int i=0;i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			for(int di=0;di<m_dispLevels; di++)
			{
				m_sgmCostCube->at<double>(di,i,j) +=  m_Lr0->at<double>(di,i,j)
					+ m_Lr1->at<double>(di,i,j)
					+ m_Lr2->at<double>(di,i,j)
					+ m_Lr3->at<double>(di,i,j);
			}
		}
	}
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			//if (useMask && m_maskImg.at<uchar>(i,j) == 0) continue;

			double min = 1e20;
			int d_min = 0;
			for(int di=0; di<m_dispLevels; di++)
			{
				double cost_t = m_sgmCostCube->at<double>(di,i,j);
				if (cost_t < min)
				{
					min = cost_t;
					d_min = di;
				}
			}
			m_dispImg.at<double>(i,j) = d_min + m_dispMin;//m_dispImg.at<double>(i,j) = d_min;
		}
	}

}

void SGM::sgmRun()
{
	rawCostCalculate();
	sgmCostCalculate();
}


void SGM::dispFromMask()
{
	int W = m_maskImgL.cols;
	int H = m_maskImgL.rows;
	//vector<vector <int> > leftMaskEdge;
	m_leftMaskEdge.resize(H);
	m_rightMaskEdge.resize(H);
	//提取左 mask 边界
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (m_maskImgL.at<uchar>(i,j) == 0) continue;
			else
			{
				m_leftMaskEdge[i].push_back(j); //左边界x
				break;
			}

		}

		for(int j=W-1; j>=0; j--)
		{
			//assert(j<W);
			if (m_maskImgL.at<uchar>(i,j) == 0) continue;
			else
			{
				m_leftMaskEdge[i].push_back(j); //右边界x
				break;
			}
		}
	}

	//提取右 mask 边界
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (m_maskImgR.at<uchar>(i,j) == 0) continue;
			else
			{
				m_rightMaskEdge[i].push_back(j);
				break;
			}

		}

		for(int j=W-1; j>=0; j--)
		{
			if (m_maskImgR.at<uchar>(i,j) == 0) continue;
			else
			{
				m_rightMaskEdge[i].push_back(j);
				break;
			}
		}
	}

	for(int i=0;i<H; i++)
	{
		if(m_leftMaskEdge[i].size()==0 || m_rightMaskEdge[i].size()==0 ) continue; //跳过全黑行

		int disp_le = abs(m_leftMaskEdge[i][0] - m_rightMaskEdge[i][0]);
		int disp_re = abs(m_leftMaskEdge[i][1] - m_rightMaskEdge[i][1]);
		m_maskEdgeDisp[i].push_back(disp_le);
		m_maskEdgeDisp[i].push_back(disp_re);

	}

	


}


void SGM::maskRawCostCalculate()
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
		
		for(int i=0; i<m_leftMaskEdge.size(); i++)
		{
			if (m_leftMaskEdge[i].size() == 0) continue; //全黑行跳过

			for(int j=m_leftMaskEdge[i][0]; j<=m_leftMaskEdge[i][1]; j++)
			{
				//if (useMask && m_maskImg.at<uchar>(i,j) == 0) continue;
				int dd = (m_maskEdgeDisp[i][0]+ m_maskEdgeDisp[i][1])/2 + (d - m_dispLevels/2);
				int disp = max(0, dd );
				/*if ( (j==m_leftMaskEdge[i][0] || j == m_leftMaskEdge[i][1])  )
				{
					m_rawCostCube->at<double>(d,i,j) = 0;
				}
				else*/
				{
					int jr = max(0,j-disp);
					//if (m_maskImgR.at<uchar>(i,jr) == 0) continue;
					m_rawCostCube->at<double>(d,i,j) = abs(( double)m_imgL.at<uchar>(i,j) - ( double)m_imgR.at<uchar>(i,jr) );
				}

				
			}
		}
	}	
		
}


void SGM::maskPathEvaluate(int x,int y, int x_r, int y_r)//void SGM::pathEvaluate(int x,int y,int d, int x_r, int y_r)
{
	//clock_t timer;static int cnt=0;if (cnt++%10000==0)timer = clock();

	for(int di=0; di<m_dispLevels; di++)
	{
		double smooth_term = 1e20;// 
		for(int d_pi=0; d_pi<m_dispLevels; d_pi++)
		{
			double priorCost = m_sgmCostCube->at<double>(d_pi,y_r,x_r);//double priorCost = m_sgmCostCube->at<double>(d,y_r,x_r);

			int d_p = (m_maskEdgeDisp[y_r][0]+ m_maskEdgeDisp[y_r][1])/2 + (d_pi - m_dispLevels/2); ///
			d_p = max(0,d_p);
			int d = (m_maskEdgeDisp[y][0]+ m_maskEdgeDisp[y][1])/2 + (di - m_dispLevels/2);
			d = max(0,d);

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
		m_sgmCostCube->at<double>(di,y,x) += m_rawCostCube->at<double>(di,y,x) + smooth_term;//m_sgmCostCube->at<double>(d,y,x) = m_rawCostCube->at<double>(d,y,x) + smooth_term;

	}


	int m = 1e20;
	for(int d_p=0; d_p<m_dispLevels; d_p++)
	{
		if (m_sgmCostCube->at<double>(d_p,y_r,x_r) < m) m = m_sgmCostCube->at<double>(d_p,y_r,x_r);
	}
	assert(m != 1e20);
	for(int dd=0; dd<m_dispLevels; dd++)
		m_sgmCostCube->at<double>(dd,y,x) -= m ;

	//if (cnt%100000 ==100000-1) cout<<clock()-timer<<endl;
}
void SGM::maskSgmCostCalculate()
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
		if (m_leftMaskEdge[y].size() == 0) continue;
		for(int x=m_leftMaskEdge[y][0]; x<=m_leftMaskEdge[y][1]; x++)
		{
			if ( m_maskImgL.at<uchar>(y,x) == 0) continue;
			//for(int d=0; d<m_dispLevels; d++)
			//{		
			int x_r=0,y_r=0;
			//if ()
			//left
			//if(x>=1)
			{
				x_r = max(x-1,0); y_r = y;
				if(m_maskImgL.at<uchar>(y_r,x_r)!=0) 
					maskPathEvaluate(x, y, x_r, y_r);
			}

			//top-left
			//if (x>=1 && y>=1)
			{
				x_r = max(x-1,0); y_r = max(y-1,0);
				if(m_maskImgL.at<uchar>(y_r,x_r)!=0) 
					maskPathEvaluate(x, y, x_r, y_r);
			}

			//top
			//if (y>=1)
			{
				x_r = x; y_r = max(y-1,0);
				if(m_maskImgL.at<uchar>(y_r,x_r)!=0) 
					maskPathEvaluate(x, y, x_r, y_r);
			}

			//top-right
			//	if (y>=1 && x<W-1)
			{
				x_r = min(x+1,W-1); y_r = max(y-1,0);
				if(m_maskImgL.at<uchar>(y_r,x_r)!=0) 
					maskPathEvaluate(x, y, x_r, y_r);
			}

			//}
		}
	}

	//从下到上
	for(int y=H-1; y>=0; y--)
	{
		if (m_leftMaskEdge[y].size() == 0) continue;
		for(int x=m_leftMaskEdge[y][1]; x>=m_leftMaskEdge[y][0]; x--)
		{
			if ( m_maskImgL.at<uchar>(y,x) == 0) continue;//if (useMask && m_maskImg.at<uchar>(y,x) == 0) continue;
			//for(int d=0; d<m_dispLevels; d++)
			//{		
			int x_r=0,y_r=0;
			//if ()
			//right
			x_r = min(x+1,W-1); y_r = y;
			if(m_maskImgL.at<uchar>(y_r,x_r)!=0)
				maskPathEvaluate(x, y,  x_r, y_r);
			//bottom-right
			x_r = min(x+1,W-1); y_r = min(y+1,H-1);
			if(m_maskImgL.at<uchar>(y_r,x_r)!=0)
				maskPathEvaluate(x, y, x_r, y_r);
			//bottom
			x_r = x; y_r = min(y+1,H-1);
			if(m_maskImgL.at<uchar>(y_r,x_r)!=0)
				maskPathEvaluate(x, y, x_r, y_r);
			//bottom-left
			x_r = max(x-1,0); y_r = min(y+1,H-1);
			if(m_maskImgL.at<uchar>(y_r,x_r)!=0)
				maskPathEvaluate(x, y, x_r, y_r);
			//}
		}
	}


	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (m_maskImgL.at<uchar>(i,j) == 0) continue;

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
			int disp = (m_maskEdgeDisp[i][0]+ m_maskEdgeDisp[i][1])/2 + (d_min - m_dispLevels/2);
			m_dispImg.at<double>(i,j) = std::max (disp,0);//d_min ;//0.5*(m_maskEdgeDisp[i][0]+ m_maskEdgeDisp[i][1]) + (d - 0.5*m_dispLevels);
		}
	}

}


void SGM:: maskSgmRun()
{
	dispFromMask();
	maskRawCostCalculate();
	maskSgmCostCalculate();
}