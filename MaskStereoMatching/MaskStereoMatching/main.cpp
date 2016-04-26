#include <iostream>
#include <fstream>
#include "time.h"

#include <cxcore.h>
#include <highgui.h>
#include "cvaux.h"
#include "cxmisc.h"

#include "SGM.h"
#include "MaskSGBM.h"
#include "ParallelMaskSGBM.h"

#include <omp.h>
using namespace std;
using namespace cv;

#if 1
int main()
{
	cv::Mat imgL = cv::imread("data4/rect_left_000035.jpg",0);//("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	cv::Mat imgR = cv::imread("data4/rect_right_000035.jpg",0);//("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat maskImgL = cv::imread("data4/mask_rect_left_000035.jpg",0);
	const cv::Mat maskImgR = cv::imread("data4/mask_rect_right_000035.jpg",0);

	int threadNum = 8;
	int dispMin=-1;
	int dispLevels=-1;
	double P1=1; double P2=1000;int filterWidth = 1;
	CParallelMaskSGBM pmsgbm(threadNum, dispMin,dispLevels,P1,P2,filterWidth );

	Mat dispImg;
clock_t timer = clock();
	pmsgbm(imgL,imgR,maskImgL, maskImgR, dispImg);
cout<<clock()-timer<<endl;
	Mat dispF;

	dispImg.convertTo(dispF,CV_64FC1);
	imwrite("dispSgmOpenCV.jpg",dispF/16);
	imshow("1",dispF/16/320);
	waitKey(0);
}
#elif 1
int main()
{
	// test class CMaskSGBM
	cv::Mat imgL = cv::imread("data4/rect_left_000035.jpg",0);//("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	cv::Mat imgR = cv::imread("data4/rect_right_000035.jpg",0);//("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat maskImgL = cv::imread("data4/mask_rect_left_000035.jpg",0);
	const cv::Mat maskImgR = cv::imread("data4/mask_rect_right_000035.jpg",0);
	Mat dispImg;
	Mat dispImg2;
	//imshow("1",maskImgR);waitKey(0);
	clock_t timer = clock();
	CMaskSGBM msgbm(-1,-1,1,1000,30);
	//msgbm(imgL,imgR,maskImgL,maskImgR,dispImg);

	//CMaskSGBM msgbm2(-1,-1,1,1000,30);
	//msgbm2(imgL,imgR,maskImgL,maskImgR,dispImg);
	//CMaskSGBM msgbm_v[] = {msgbm,msgbm2};
	//Mat * dispImg_v[] = {&dispImg,&dispImg2};

	int coreNum = omp_get_num_procs();//获得处理器个数
	cout<< coreNum<<endl;
#pragma omp parallel for
	for(int i=0; i<1; i++)
	{
		msgbm (imgL,imgR,maskImgL,maskImgR, dispImg);
	}
	
	//dispImg.convertTo(dispImg,CV_64FC1);
	//dispImg = dispImg/16/320;
	//msgbm.meanFilter(dispImg);
	//imshow("2",imgL/255);

	//msgbm.meanFilter(dispImg);
	////imshow("3",imgL/255);*/
	//
	//msgbm.meanFilter(dispImg);
	//msgbm.meanFilter(dispImg);
	//msgbm.meanFilter(dispImg);
	//msgbm.meanFilter(dispImg);
	//msgbm.meanFilter(dispImg);

	//StereoSGBM sgm(0,320,1,p1,p2);
	Mat dispF;
	//sgm(imgL,imgR,disp);
	cout<<clock()-timer<<endl;

	dispImg.convertTo(dispF,CV_64FC1);
	imwrite("dispSgmOpenCV.jpg",dispF/16);
	ofstream ofile("disp.txt");
	ofile<<dispF.rows<<" "<<dispF.cols<<endl;
	for(int i=0; i<dispF.rows; i++)
	{
		for(int j=0; j<dispF.cols; j++)
		{
			ofile<<dispF.at<double>(i,j)/16/300<<' ';
		}
	}
	imshow("1",(dispF)/320/16);
	waitKey(0);
}
#elif 1
int main()
{
	char root[] = "e:/MyDocument/klive sync/计算机视觉/Stereo_Matching/Data Set/Middlebury2006/half size/data";
	char dirL[100];
	char dirR[100];
	sprintf(dirL,"%s%d%s",root,2,"/view1.png");
	sprintf(dirR,"%s%d%s",root,2,"/view5.png");

	const cv::Mat imgL = cv::imread("data4/rect_left_000035t.jpg",0);//("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread("data4/rect_right_000035t.jpg",0);//("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat maskImgL = cv::imread("data4/mask_rect_left_000035.jpg",0);
	const cv::Mat maskImgR = cv::imread("data4/mask_rect_right_000035.jpg",0);
	Mat dispImg(imgL.size(),CV_64FC1,Scalar(0));

	clock_t timer = clock();
	SGM sgm(imgL,imgR,dispImg,30,70,2,1000);//SGM sgm(imgL,imgR,dispImg,130,250,2,1000);//SGM sgm(imgL,imgR,maskImgL,maskImgR,dispImg,100,10,1000);//SGM sgm(imgL,imgR,dispImg,50,2,100);//SGM sgm(imgL,imgR,dispImg,20,2,100);
	sgm.sgmRun(); //sgm.maskSgmRun(); //sgm.sgmRun(); 
	cout<<clock()-timer<<endl;

	imwrite("data4/disp.jpg",dispImg);
	imshow("1",imgL);
	imshow("2",dispImg/100);
	waitKey(0);
}
#elif	1	
int main()
{
	//opencv stereoSgbm test

	const cv::Mat imgL = cv::imread ("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);//("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat maskImgL = cv::imread("data4/mask_rect_left_000035.jpg",0);
	const cv::Mat maskImgR = cv::imread("data4/mask_rect_right_000035.jpg",0);

	int p1,p2;
	p1=2;p2=1000;
clock_t timer = clock();
	StereoSGBM sgm(0,320,1,p1,p2);
	Mat disp,dispF;
	sgm(imgL,imgR,disp);
cout<<clock()-timer<<endl;
	disp.convertTo(dispF,CV_64FC1);
	imwrite("dispSgmOpenCV.jpg",dispF/16);
	imshow("1",dispF/320/16);
	waitKey(0);

}
#elif 1
int main()
{
	char root[] = "e:/MyDocument/klive sync/计算机视觉/Stereo_Matching/Data Set/Middlebury2006/half size/data";
	char dirL[100];
	char dirR[100];
	sprintf(dirL,"%s%d%s",root,2,"/view1.png");
	sprintf(dirR,"%s%d%s",root,2,"/view5.png");

	const cv::Mat imgL = cv::imread( "./data1/im0.ppm", 0);//("data4/rect_left_000035.jpg",0);//("./data1/im0.ppm", 0);//(dirL, 0);//("./data/scene1.row3.col3.ppm", 0);//("./data2/view1_half.png", 0);//("./data/scene1.row3.col3.ppm", 0); //Load as grayscale
	const cv::Mat imgR = cv::imread ("./data1/im1.ppm", 0);//("data4/rect_right_000035.jpg",0);//("./data1/im1.ppm", 0);// (dirR, 0);//("./data/scene1.row3.col4.ppm", 0);
	const cv::Mat maskImgL = cv::imread("data4/mask_rect_left_000035.jpg",0);
	const cv::Mat maskImgR = cv::imread("data4/mask_rect_right_000035.jpg",0);
	Mat dispImg(imgL.size(),CV_64FC1,Scalar(0));
	
clock_t timer = clock();
	SGM sgm(imgL,imgR,dispImg,30,2,1000);//SGM sgm(imgL,imgR,maskImgL,maskImgR,dispImg,100,10,1000);//SGM sgm(imgL,imgR,dispImg,50,2,100);//SGM sgm(imgL,imgR,dispImg,20,2,100);
	sgm.sgmRun(); //sgm.maskSgmRun(); //sgm.sgmRun(); 
cout<<clock()-timer<<endl;

	//imwrite("data4/disp.jpg",dispImg);
	imshow("1",imgL);
	imshow("2",dispImg/30);
	waitKey(0);
}
#elif 1
static int curNo = 0;
float mp[924][517][100];
int main()
{
	for (int i = 0; i< 924; i++)
		for(int j = 0; j < 517; j++)
			for (int d = 0; d < 100; d++)
				mp[i][j][d] = 0;
	const int dsp = 10;
	vector<vector<pair<int,int>>> maskVL, maskVR;
	Mat maskL = imread("M0001L.PNG"),maskR = imread("M0001R.PNG");
	Mat imgL = imread("0001L.PNG"), imgR = imread("0001R.PNG");
	cvtColor(maskL, maskL, CV_BGR2GRAY);
	cvtColor(maskR, maskR, CV_BGR2GRAY);
	int w = maskL.cols, h = maskL.rows;
	maskVL.resize(h); maskVR.resize(h);
	ZFloatImage datacostArr(w,h,dsp*2);
	for(int i =0; i < w; i++)
		for(int j = 0; j < h; j++)
			for(int k = 0; k < dsp*2; k++)
				datacostArr.at(i,j,k) = -1;
	//得到只有前景的左右图片
	//for (int x = 0; x < w; x++)
	//	for (int y = 0; y < h; y++)
	//	{
	//		if (maskL.at<uchar>(y,x) == 0)
	//		{
	//			imgL.at<Vec3b>(y,x)[0] = imgL.at<Vec3b>(y,x)[1] = imgL.at<Vec3b>(y,x)[2] = 0.0;
	//		}
	//		if (maskR.at<uchar>(y,x) == 0)
	//		{
	//			imgR.at<Vec3b>(y,x)[0] = imgR.at<Vec3b>(y,x)[1] = imgR.at<Vec3b>(y,x)[2] = 0.0;
	//		}
	//	}
	//	imwrite("R.jpg",imgR);
	//	imwrite("L.jpg",imgL);

	//对左图片进行处理

	cout<<h<<" ** "<<w<<endl;
	for (int v = 0; v < h; v++)
	{
		int s = 0, e = 0; bool isBlack = true;
		vector<pair<int,int>> tmp;
		for(int u = 0; u < w; u++)
		{
			int  gray = (int)maskL.at<uchar>(v, u);
			//if(gray!=0)cout<<u<<" "<<v<<" "<<gray<<endl;
			if(isBlack)
			{
				if(gray == 255)
				{
					//cout<<"*"<<endl;
					s = u; isBlack = false;
				}
			}
			else if(!isBlack)
			{
				if(gray == 0)
				{
					e = u; isBlack = true;
					//cout<<"push"<<s<<" "<<e<<endl;
					tmp.push_back(make_pair(s,e));
				}
			}
		}
		if(!tmp.empty())
		{
			s = tmp[0].first; e = tmp[tmp.size()-1].second;
			tmp.clear(); tmp.push_back(make_pair(s,e));
		}
		if(!tmp.empty())
			maskVL[v] = tmp;
	}

	//imshow("L",maskL);
	//imshow("R",maskR);
	//waitKey(0);
	const int Threshd = 20;
	for (int i = 0; i < maskVL.size(); i++)
	{
		if(maskVL[i].empty()) continue;
		else
		{
			int s = 0 , e = 0;
			for (int j = 0; j < maskVL[i].size(); j++)
			{
				pair<int,int> p = maskVL[i][j];
				//处理小小的白片区域
				if(p.second - p.first < Threshd) continue;
				else
				{
					//处理中间的小黑片;
					if(p.first - e < Threshd) e = p.second;

					//正常情况下，取最大的间隔区域;
					else if(p.second - p.first > e - s){s = p.first; e = p.second;}
				}
			}
			if(s != 0 && e != 0)maskVL[i][0].first = s; maskVL[i][0].second = e;
			while (maskVL[i].size() != 1) maskVL[i].pop_back();
		}
	}

	//对右图片进行处理;
	for (int v = 0; v < h; v++)
	{
		int s = 0, e = 0; bool isBlack = true;
		vector<pair<int,int>> tmp;
		for(int u = 0; u < w; u++)
		{
			int  gray = (int)maskR.at<uchar>(v, u);
			if(isBlack)
			{
				if(gray == 255)
				{s = u; isBlack = false;}
			}
			else if(!isBlack)
			{
				if(gray == 0)
				{e = u; isBlack = true; tmp.push_back(make_pair(s,e));}
			}
		}
		if(!tmp.empty())
		{
			s = tmp[0].first; e = tmp[tmp.size()-1].second;
			tmp.clear(); tmp.push_back(make_pair(s,e));
		}
		if(!tmp.empty())
			maskVR[v] = tmp;
	}

	for (int i = 0; i < maskVR.size(); i++)
	{
		if(maskVR[i].empty()) continue;
		else
		{
			int s = 0 , e = 0;
			for (int j = 0; j < maskVR[i].size(); j++)
			{
				pair<int,int> p = maskVR[i][j];
				//处理小小的白片区域
				if(p.second - p.first < Threshd) continue;
				else
				{
					//处理中间的小黑片;
					if(p.first - e < Threshd) e = p.second;

					//正常情况下，取最大的间隔区域;
					else if(p.second - p.first > e - s){s = p.first; e = p.second;}
				}
			}
			if(s != 0 && e != 0)maskVR[i][0].first = s; maskVR[i][0].second = e;
			while (maskVR[i].size() != 1) maskVR[i].pop_back();
		}
	}
	clock_t time = clock();
	Mat disparity(h,w,CV_8UC1);
	//对左右mask进行匹配



	//multimap<pair<int,int>,pair<int,float>> mp;
	int p1 = 2, p2 =10;//1000;
	ZFloatImage minCost;
	minCost.CreateAndInit(w,h,1,0);
	int low = 0, high = 0;
	for (int i = 0; i < maskVL.size(); i++)
	{
		if(maskVL[i].empty() || maskVL[i][0].second < maskVL[i][0].first || maskVR[i][0].second < maskVR[i][0].first) continue;
		else
		{
			for (int j = maskVL[i][0].first; j <= maskVL[i][0].second; j++)
			{
				float tmpMin = INT_MAX;
				if(j == maskVL[i][0].first)
				{
					disparity.at<uchar>(i,j) = maskVR[i][0].first - maskVL[i][0].first;
				}

				else if(j == maskVL[i][0].second)
				{
					disparity.at<uchar>(i,j) = maskVR[i][0].second - maskVL[i][0].second;
				}
				//对于非边界区域利用dp scanline
				else
				{
					if( low == 0 && high == 0)
					{
						low = min(maskVR[i][0].first - maskVL[i][0].first,maskVR[i][0].second - maskVL[i][0].second);
						high = max(maskVR[i][0].first - maskVL[i][0].first,maskVR[i][0].second - maskVL[i][0].second);
					}
					double datacost = INT_MAX;
					int lowerBound = max(low - dsp,0);
					for (int idx = lowerBound; idx < min(high+dsp,w); idx++)
					{
						float cost = abs(imgL.at<Vec3b>(i,j)[0] - imgR.at<Vec3b>(i,j+idx)[0]) + abs(imgL.at<Vec3b>(i,j)[1] - imgR.at<Vec3b>(i,j+idx)[1]) + abs(imgL.at<Vec3b>(i,j)[2] - imgR.at<Vec3b>(i,j+idx)[2]);

						if(minCost.at(j-1,i) > 0.01) {
							double tmp = cost;
							for (int pp = max(low - dsp,0); pp <  min(high+dsp,w); pp++)
							{
								if(pp == idx) tmp = cost + mp[j-1][i][idx-lowerBound] - minCost.at(j-1,i);
								else if(abs(pp-idx) == 1) tmp = cost + mp[j-1][i][idx-lowerBound] - minCost.at(j-1,i) + p1;
								else
								{
									//cout<<j<<" "<<i<<" "<<idx<<" "<<lowerBound<<endl;
									if(abs(cost < 0.01)) tmp = cost + mp[j-1][i][idx-lowerBound] - minCost.at(j-1,i) + p2;

									else tmp = cost + mp[j-1][i][idx-lowerBound] - minCost.at(j-1,i) + p2/cost;
								}
							}
							cost = tmp;
						}

						if(minCost.at(j,i-1) > 0.01) {
							double tmp = cost;
							for (int pp = max(low - dsp,0); pp <  min(high+dsp,w); pp++)
							{
								if(pp == idx) tmp = cost + mp[j][i-1][idx-lowerBound] - minCost.at(j,i-1);
								else if(abs(pp-idx) == 1) tmp = cost + mp[j][i-1][idx-lowerBound] - minCost.at(j,i-1) + p1;
								else
								{
									//cout<<j<<" "<<i<<" "<<idx<<" "<<lowerBound<<endl;
									if(abs(cost < 0.01)) tmp = cost + mp[j][i-1][idx-lowerBound] - minCost.at(j,i-1) + p2;

									else tmp = cost + mp[j][i-1][idx-lowerBound] - minCost.at(j,i-1) + p2/cost;
								}
							}
							cost = tmp;
						}

						mp[j][i][idx-lowerBound] = cost;
						if(tmpMin > cost)
						{
							tmpMin = cost;
							disparity.at<uchar>(i,j) = idx;
							minCost.at(j,i) = cost;
						}
					}
				}
			}
		}
	}

	//从下往上
	for (int i = maskVL.size() - 1; i >= 0; i--)
	{
		if(maskVL[i].empty() || maskVL[i][0].second < maskVL[i][0].first || maskVR[i][0].second < maskVR[i][0].first) continue;
		else
		{
			for (int j = maskVL[i][0].second; j >= maskVL[i][0].first; j--)
			{
				float tmpMin = INT_MAX;
				if(j != maskVL[i][0].first && j != maskVL[i][0].second)
				{
					int lowerBound = max(low - dsp,0);
					for (int idx = lowerBound; idx < min(high+dsp,w); idx++)
					{
						float cost = mp[j][i][idx-lowerBound];
						if(minCost.at(j,i+1) > 0.01) {
							double tmp = cost;
							for (int pp = max(low - dsp,0); pp <  min(high+dsp,w); pp++)
							{
								if(pp == idx) tmp = cost + mp[j][i+1][idx-lowerBound] - minCost.at(j,i+1);
								else if(abs(pp-idx) == 1) tmp = cost + mp[j][i+1][idx-lowerBound] - minCost.at(j,i+1) + p1;
								else
								{
									//cout<<j<<" "<<i<<" "<<idx<<" "<<lowerBound<<endl;
									if(abs(cost < 0.01)) tmp = cost + mp[j][i+1][idx-lowerBound] - minCost.at(j,i+1) + p2;

									else tmp = cost + mp[j][i+1][idx-lowerBound] - minCost.at(j,i+1) + p2/cost;
								}
							}
							cost = tmp;
						}

						//if(minCost.at(j+1,i) > 0.01) {
						//	double tmp = cost;
						//	for (int pp = max(low - dsp,0); pp <  min(high+dsp,w); pp++)
						//	{
						//		if(pp == idx) tmp = cost + mp[j+1][i][idx-lowerBound] - minCost.at(j+1,i);
						//		else if(abs(pp-idx) == 1) tmp = cost + mp[j+1][i][idx-lowerBound] - minCost.at(j+1,i) + p1;
						//		else
						//		{
						//			if(abs(cost < 0.01)) tmp = cost + mp[j+1][i][idx-lowerBound] - minCost.at(j+1,i) + p2;

						//			else tmp = cost + mp[j+1][i][idx-lowerBound] - minCost.at(j+1,i) + p2/cost;
						//		}
						//	}
						//	cost = tmp;
						//}

						mp[j][i][idx-lowerBound] = cost;
						if(tmpMin > cost)
						{
							//cout<<cost<<endl;
							tmpMin = cost;
							disparity.at<uchar>(i,j) = idx;
							minCost.at(j,i) = cost;
						}
					}
				}
			}
		}
	}

	for (int x = 0; x < w; x++)
		for (int y = 0; y < h; y++)
		{
			if (maskL.at<uchar>(y,x) < 50)
			{
				disparity.at<uchar>(y,x) = 0;
			}
			else disparity.at<uchar>(y,x) *= 2;
		}

		//	medianBlur(disparity,disparity,5);
		printf("%f\n",(double)(clock() - time)/CLOCKS_PER_SEC);

		imwrite("disparity.jpg",disparity);

		return 0;
		//if(argc!=6)
		//{
		//	printf("Usage:\n");
		//	printf("*.exe: filename_disparity_map filename_left_image filename_right_image max_disparity\n");
		//	return(-1);
		//}
		//char filename_disparity_map[200],filename_left_image[200],filename_right_image[200];
		//int max_disparity=atoi(argv[4]);
		//int imgLen = atoi(argv[5]);
		//////init
		////for (int i = 0; i < imgLen; i++)
		////{
		////	sprintf(filename_disparity_map,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\res\\dsp%04d.pgm",i);
		////	sprintf(filename_left_image,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\rect_left_%06d.ppm",i);
		////	sprintf(filename_right_image,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\rect_right_%06d.ppm",i);
		////    stereo(filename_disparity_map,filename_left_image,filename_right_image,max_disparity,0,i);//including non-local post processing
		////}
		////refine
		//for (int i = 0; i < imgLen - 5; i++)
		//{
		//	curNo = i;
		//	sprintf(filename_disparity_map,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\res\\dsp%04d.pgm",i);
		//	sprintf(filename_left_image,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\rect_left_%06d.ppm",i);
		//	sprintf(filename_right_image,"C:\\Users\\itsuhane\\Documents\\Binocular\\VS2008\\data9\\rect_right_%06d.ppm",i);
		//	stereo_refine(filename_disparity_map,filename_left_image,filename_right_image,max_disparity,0);//including non-local post processing
		//}

		return(0);
}

#endif