#pragma once

#include "cxcore.h"
class SGM
{
private:
	cv::Mat m_imgL,m_imgR;
	cv::Mat & m_dispImg;
	cv::Mat * m_rawCostCube;
	cv::Mat * m_sgmCostCube;
	int m_dispLevels;
	double m_P1, m_P2;

	inline void pathEvaluate(int x,int y,int x_r, int y_r);//void pathEvaluate(int x,int y,int d,int x_r, int y_r);
	void rawCostCalculate();
	void sgmCostCalculate();
public:
	
	SGM();
	SGM(const cv::Mat &imgL,const cv::Mat &imgR,cv::Mat & dispImg,int dispLevels = 20,double P1=1,double P2=500)
		:m_dispImg(dispImg),m_dispLevels(dispLevels),m_P1(P1),m_P2(P2)
	{
		m_imgL = imgL.clone();
		m_imgR = imgR.clone();
		m_rawCostCube = NULL;
		m_sgmCostCube = NULL;
		//cv::Mat m_dispImg(imgL.size(),CV_64FC1,Scalar(0));
		//m_dispImg = dispImg;
	}
	void sgmRun();
	
};