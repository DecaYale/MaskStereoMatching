#pragma once

#include "cxcore.h"
class SGM
{
private:
	cv::Mat m_imgL,m_imgR;
	cv::Mat m_maskImgL;
	cv::Mat m_maskImgR;
	cv::Mat & m_dispImg;
	cv::Mat * m_rawCostCube;
	cv::Mat * m_sgmCostCube;
	cv::Mat * m_Lr0;
	cv::Mat * m_Lr1;
	cv::Mat * m_Lr2;
	cv::Mat * m_Lr3;

	std::vector<std::vector <int> > m_leftMaskEdge; //saving the x location of L&R side of the mask in m_maskImgL;
	std::vector<std::vector <int> > m_rightMaskEdge;
	std::vector<std::vector <int> > m_maskEdgeDisp;
	int m_dispLevels;
	double m_P1, m_P2;

	inline void pathEvaluate(int x,int y,int x_r, int y_r);//void pathEvaluate(int x,int y,int d,int x_r, int y_r);
	void rawCostCalculate();
	void sgmCostCalculate();
	void dispFromMask();
	void maskRawCostCalculate();
	void maskSgmCostCalculate();
	inline void maskPathEvaluate(int x,int y, int x_r, int y_r);
	inline void pathEvaluate(int x,int y, int x_r, int y_r,cv::Mat * Lr);
public:
	
	SGM();
	SGM(const cv::Mat &imgL,const cv::Mat &imgR,cv::Mat & dispImg,int dispLevels = 20,double P1=1,double P2=500)
		:m_dispImg(dispImg),m_dispLevels(dispLevels),m_P1(P1),m_P2(P2)
	{
		m_imgL = imgL.clone();
		m_imgR = imgR.clone();
		m_rawCostCube = NULL;
		m_sgmCostCube = NULL;
		m_Lr0 = NULL;
		m_Lr1 = NULL;
		m_Lr2 = NULL;
		m_Lr3 = NULL;
		//cv::Mat m_dispImg(imgL.size(),CV_64FC1,Scalar(0));
		//m_dispImg = dispImg;
	}
	SGM(const cv::Mat &imgL,const cv::Mat &imgR,const cv::Mat & maskImgL,const cv::Mat &maskImgR, cv::Mat & dispImg,int dispLevels = 20,double P1=1,double P2=500)
		:m_dispImg(dispImg),m_dispLevels(dispLevels),m_P1(P1),m_P2(P2)
	{
		m_imgL = imgL.clone();
		m_imgR = imgR.clone();
		m_maskImgL = maskImgL.clone();
		m_maskImgR = maskImgR.clone();
		m_leftMaskEdge.resize(m_maskImgL.rows);
		m_rightMaskEdge.resize(m_maskImgR.rows);
		m_maskEdgeDisp.resize(m_maskImgR.rows);

		m_rawCostCube = NULL;
		m_sgmCostCube = NULL;
		//cv::Mat m_dispImg(imgL.size(),CV_64FC1,Scalar(0));
		//m_dispImg = dispImg;
	}
	void sgmRun();
	void maskSgmRun();
	//void maskSgmRun();
};