#include "cxcore.h"
#include "MaskSGBM.h"
#include "highgui.h"

#include "ParallelMaskSGBM.h"
using namespace cv;


void CParallelMaskSGBM::operator() (const cv::Mat & imgL, const cv::Mat &imgR, const cv::Mat maskImgL, const cv::Mat maskImgR, cv::Mat &dispImg)
{
	int H = imgL.rows;
	int W = imgL.cols;
	m_block_H = H / m_blockNum  + 1;
	
	int overlap = m_block_H*0.25;

	Mat * dispImgBuf = new Mat [m_blockNum];
	//decompose
#pragma omp parallel for  //num_threads(1)
	for(int i=0; i<m_blockNum; i++)
	{
		Mat imgL_i = imgL.rowRange(
									max(0,i*m_block_H - overlap), min(H-1,(i+1)*m_block_H + overlap)  
								  );
		Mat imgR_i = imgR.rowRange(
									max(0,i*m_block_H - overlap), min(H-1,(i+1)*m_block_H + overlap)  
								);
		Mat maskImgL_i = maskImgL.rowRange(
											max(0,i*m_block_H - overlap), min(H-1,(i+1)*m_block_H + overlap)  
											);
		Mat maskImgR_i = maskImgR.rowRange(
											max(0,i*m_block_H - overlap), min(H-1,(i+1)*m_block_H + overlap)  
											);
		

		m_MaskSGBM_BUF[i](imgL_i, imgR_i, maskImgL_i, maskImgR_i, dispImgBuf[i]);

	}

	//compose
	dispImg.create(imgL.size(),CV_16SC1);
	for(int y=0; y<H; y++)
	{
		int i = y / m_block_H;
		int yi = y % m_block_H;

		for(int x=0; x<W; x++)
		{
			dispImg.at<short int>(y,x) = dispImgBuf[i].at<short int>(yi,x);
		}
		
	}
	



}
void meanFilter(cv::Mat & dispImg);
