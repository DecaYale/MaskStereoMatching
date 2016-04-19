#include "cxcore.h"
#include "MaskSGBM.h"
#include "highgui.h"
using namespace cv;



void CMaskSGBM::getMaskROI(const cv::Mat & maskImgL, const cv::Mat & maskImgR )
{
	int W = maskImgL.cols;
	int H = maskImgL.rows;
	//vector<vector <int> > leftMaskEdge;
	m_leftMaskEdge.resize(H);
	m_rightMaskEdge.resize(H);
	//提取左 mask 边界
	for(int i=0; i<H; i++)
	{
		for(int j=0; j<W; j++)
		{
			if (maskImgL.at<uchar>(i,j) == 255) 
			{
				m_leftMaskEdge[i].push_back(j); //左边界x
				break;
			}

		}

		for(int j=W-1; j>=0; j--)
		{
			//assert(j<W);
			if (maskImgL.at<uchar>(i,j) == 255) 
			{
				//assert(j!=991);
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
			if (maskImgR.at<uchar>(i,j) == 255)
			{
				//assert(j!=991 && m_rightMaskEdge[i].size()==0);
				m_rightMaskEdge[i].push_back(j);
				break;
			}

		}
		//std::cout<<(int)maskImgR.at<uchar>(i,991)<<"\n";
		for(int j=W-1; j>=0; j--)
		{

			if (maskImgR.at<uchar>(i,j) == 255) 
			{
				//assert(j!=991 && m_rightMaskEdge[i].size()==0);
				if (j==991) std::cout<<i<<' '<<(int)maskImgR.at<uchar>(i,j)<<"\n";
				m_rightMaskEdge[i].push_back(j);
				break;
			}
		}
	}
	//求左图像ROI
	int ltx = 1e20,lty=0;
	int rbx = 0,rby=0;
	for(int i=0; i<H; i++)
	{
		if (m_leftMaskEdge[i].size()!=0)
		{
			lty = i;
			break;
		}
	}
	for(int i=H-1; i>=0; i--)
	{
		if (m_leftMaskEdge[i].size()!=0)
		{
			rby = i;
			break;
		}
	}
	for(int i=0; i<H; i++)
	{
		if (m_leftMaskEdge[i].size()!=0)
		{
			if (m_leftMaskEdge[i].at(0) < ltx) ltx = m_leftMaskEdge[i].at(0);
		}
	}
	for(int i=0; i<H; i++)
	{
		if (m_leftMaskEdge[i].size()!=0)
		{
			if (m_leftMaskEdge[i].at(1) > rbx) rbx = m_leftMaskEdge[i].at(1);
		}
	}


	//求右图像ROI
	int ltx_r = 1e20,lty_r=0;
	int rbx_r = 0,rby_r=0;
	for(int i=0; i<H; i++)
	{
		if (m_rightMaskEdge[i].size()!=0)
		{
			lty_r = i;
			break;
		}
	}
	for(int i=H-1; i>=0; i--)
	{
		if (m_rightMaskEdge[i].size()!=0)
		{
			rby_r = i;
			break;
		}
	}
	for(int i=0; i<H; i++)
	{
		if (m_rightMaskEdge[i].size()!=0)
		{
			if (m_rightMaskEdge[i].at(0) < ltx_r) ltx_r = m_rightMaskEdge[i].at(0);
		}
	}
	for(int i=0; i<H; i++)
	{
		if (m_rightMaskEdge[i].size()!=0)
		{
			if (m_rightMaskEdge[i].at(1) > rbx_r) rbx_r = m_rightMaskEdge[i].at(1);
		}
	}


	/*m_ImgLROI.y = min(lty, lty_r);		m_ImgRROI.y = m_ImgLROI.y;
	m_ImgLROI.x = ltx;					m_ImgRROI.x = ltx_r;

	m_ImgLROI.width = max( abs(ltx - rbx), abs(ltx_r - rbx_r) ); m_ImgRROI.width = m_ImgLROI.width;
	m_ImgLROI.height = max(abs(lty - rby), abs(lty_r - rby_r) ); m_ImgRROI.height = m_ImgLROI.height;*/
	int x = min(ltx, ltx_r);
	int y = min(lty, lty_r);
	int x2 = max(rbx,rbx_r);
	int y2 = max(rby,rby_r);
	int width = x2-x;//max( abs(ltx - rbx), abs(ltx_r - rbx_r) );
	int height = y2-y;//max(abs(lty - rby), abs(lty_r - rby_r) ); 
	m_ImgLROI = cv::Rect(x,y,width,height);
	m_ImgRROI = cv::Rect(x,y,width,height);



	//for(int i=0;i<H; i++)
	//{
	//	if(m_leftMaskEdge[i].size()==0 || m_rightMaskEdge[i].size()==0 ) continue; //跳过全黑行

	//	int disp_le = abs(m_leftMaskEdge[i][0] - m_rightMaskEdge[i][0]);
	//	int disp_re = abs(m_leftMaskEdge[i][1] - m_rightMaskEdge[i][1]);
	//	maskEdgeDisp[i].push_back(disp_le);
	//	maskEdgeDisp[i].push_back(disp_re);

	//}

}

void CMaskSGBM::getDispFromROI(const cv::Mat & imgL, const cv::Mat &imgR, cv::Mat & dispImg)
{
	imshow("tmp",imgL(m_ImgLROI));imshow("tmp2",imgR(m_ImgRROI));
	m_sgbm(imgL(m_ImgLROI), imgR(m_ImgRROI), dispImg);
}
void CMaskSGBM::operator() (const cv::Mat & imgL, const cv::Mat &imgR, const cv::Mat maskImgL, const cv::Mat maskImgR, cv::Mat & dispImg)
{
	getMaskROI(maskImgL, maskImgR );
	getDispFromROI(imgL, imgR, dispImg);
}

