#pragma once

#include <vector>
#include "cxcore.h"
#include "cvaux.h"
#include "cxmisc.h"

#include "MaskSGBM.h"
class CParallelMaskSGBM
{
private:
	CMaskSGBM * m_MaskSGBM_BUF;
	int m_blockNum;
	int m_dispMin;
	int m_dispLevels;
	int m_P1;
	int m_P2;
	int m_filterWidth;
	int m_block_H;
public:
	//CParallelMaskSGBM(){};
	CParallelMaskSGBM(int threadNum=1, int dispMin=-1,int dispLevels=-1, double P1=0, double P2=0,int filterWidth = 1)  //int dispMin=-1,int dispLevels=-1则放弃mask动态disp范围
		:m_blockNum(threadNum),m_dispMin(dispMin), m_dispLevels(dispLevels), m_P1(P1), m_P2(P2), m_filterWidth(filterWidth)
	{
		m_MaskSGBM_BUF = new CMaskSGBM[threadNum];
		for(int i=0; i<threadNum; i++)
		{
			m_MaskSGBM_BUF[i] = CMaskSGBM(dispMin,dispLevels, P1, P2 ,filterWidth); //int dispMin=-1,int dispLevels=-1则放弃mask动态disp范围	
		}
	}
	~CParallelMaskSGBM()
	{
		delete [] m_MaskSGBM_BUF;
	}
	void operator() (const cv::Mat & imgL, const cv::Mat &imgR, const cv::Mat maskImgL, const cv::Mat maskImgR, cv::Mat &dispImg);
	void meanFilter(cv::Mat & dispImg);

};