#include <vector>
#include "cxcore.h"
#include "cvaux.h"
#include "cxmisc.h"
class CMaskSGBM
{
private:
	cv::Mat m_maskImgL;
	cv::Mat m_maskImgR;
	//cv::Mat & m_dispImg;
	cv::StereoSGBM m_sgbm;

	int m_dispMin;
	int m_dispLevels;
	double m_P1, m_P2;
	cv::Rect m_ImgLROI;
	cv::Rect m_ImgRROI;
	std::vector<std::vector <int> > m_leftMaskEdge; //saving the x location of L&R side of the mask in m_maskImgL;
	std::vector<std::vector <int> > m_rightMaskEdge;
	std::vector<std::vector <int> > m_maskEdgeDisp;
public:
	CMaskSGBM(){};
	CMaskSGBM(int dispMin,int dispLevels, double P1, double P2)
			:m_dispMin(dispMin), m_dispLevels(dispLevels), m_P1(P1), m_P2(P2)
	{
		m_sgbm = cv::StereoSGBM(m_dispMin,m_dispLevels,1,m_P1, m_P2);
	}
	void operator() (const cv::Mat & imgL, const cv::Mat &imgR, const cv::Mat maskImgL, const cv::Mat maskImgR, cv::Mat &dispImg);
	void getMaskROI(const cv::Mat & maskImgL, const cv::Mat & maskImgR );
	void getDispFromROI(const cv::Mat & imgL, const cv::Mat &imgR, cv::Mat & dispImg);
};