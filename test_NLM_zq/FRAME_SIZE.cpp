////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	Program Description:	A comparison of PSNR with and without using Non-Local Means algorithm
//	Author:		ZhangQing
//	Date:		2016/5/31
//	Function:	Non_Local_Means: main implementation
//								Filtered saving results
//								Rec,Pred are buffers contain reconstruction pixels and prediction pixels
//				calcPSNR：		calculate PSNR of two inputs 	
///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <iterator>
//#include <string>
//#include <sstream>
using namespace cv;
using namespace std;

#define TemplateWindowSize 3
#define SearchWindowSize 7
#define PrintReconstructionError 0
#define NLM_Filter 1

//定义全局变量
string origfile_name, recfilename, result_name;
int WIDTH, HEIGHT, FRAME_NUMER, FRAME_SIZE;
double noise_sigma = 0.0;
//zq 8-1
const int NumPatch = 10;


int pos(int row, int col, int width){
	return row * width + col;
}

void Non_Local_Means(uchar* Src, uchar* Dest, int width, int height, int templateWindowSize, int searchWindowSize, double h, double sigma = 0.0)
{
	if(templateWindowSize > searchWindowSize)
	{
		cout<<"searchWindowSize should be larger than templateWindowSize"<<endl;
		return;
	}
	
	for(int j = 0; j < height; j++)
		for(int i = 0; i < width; i++)
			Dest[j * width + i] = Src[j * width + i];

	//Parameter Calculation
	const int tr = templateWindowSize >> 1;
	const int sr = searchWindowSize >> 1;
	const int bb = sr + tr;
	const int D = searchWindowSize * searchWindowSize;
	const int tD = templateWindowSize * templateWindowSize;
	const int H = D/2 + 1;
	const double tdiv = 1.0 / (double)(tD);

	//8.1 zq
	map<double, int> optimal;

	//	Weight computation
	vector<double> weight(256 * 256);

	double* w = &weight[0];
	const double gauss_sd = (sigma == 0.0) ? h : sigma;
	double gauss_color_coeff = - (1.0/(h * h));
	int emax;
	for(int i = 0; i < 256 * 256; i++)
	{
		double v = std::exp( max(i-2.0*gauss_sd*gauss_sd, 0.0) * gauss_color_coeff);
		w[i] = v;

		if(v < 0.0001)
		{
			emax = i;
			break;
		}
	}
	for(int i = emax; i < 256 * 256; i++) w[i] = 0.0;

	//Search loop & Template loop
	const int cstep = width - templateWindowSize;
	const int csstep = width - searchWindowSize;
#pragma omp parallel for

	for(int j = 0; j < (height - 2 * bb); j++)
	{
		//char* d = Dest + j * width;
		int Offset;
		int* ww = new int[D];
		double* nw = new double[D];
		for(int i = 0; i < (width - 2 * bb); i++)
		{
			double tweight = 0.0;
			//search loop
			uchar* tprt = Src + width * (sr + j) + (sr + i);
			uchar* sptr2 = Src + width * j + i;
			for(int l = searchWindowSize, count = D; l--;)
			{
				uchar* sptr = sptr2 + width * l;
				for(int k = searchWindowSize; k--;)
				{

					//template loop
					int e = 0;
					uchar* t = tprt;
					uchar* s = sptr + k;
					for(int n = templateWindowSize; n--;)
					{
						for(int m = templateWindowSize; m--;)
						{					
							e += (*s - *t) * (*s - *t);
							s ++;
							t ++;
						}
						s += cstep;
						t += cstep;
					}

					const int ediv = (int)e * tdiv;
					count--;
					ww[count] = ediv;
					tweight += w[ediv];// Z(i)
				}
			}
			optimal.clear();
			if(tweight == 0.0)
			{
				for(int z = 0; z < D; z++) nw[z] = 0;
				nw[H] = 1;
			}
			else
			{
				int wwz;
				double itweight = 1.0 / (double)tweight;//  1/Z(i)
				for(int z = 0; z < D; z++)
				{
					wwz = ww[z];
					nw[z] = w[wwz] * itweight;
				}
			}

			double v = 0.0;
			uchar* s = Src + (j + tr) * width;
			s += (i + tr);
			for(int l = searchWindowSize, count = 0; l--;)
			{
				for(int k = searchWindowSize; k--;)
				{
					v += *(s++) * nw[count++];
				}
				s += csstep;
			}
			Offset = (j + bb) * width + (i + bb);
			//*(d++) = saturate_cast<uchar>(v);
			Dest[Offset] = saturate_cast<uchar>(v);

		}//i
		delete[] ww;
		delete[] nw;
	}//j
}
double calcPSNR(uchar* Ori, uchar* Result, int width, int height)
{
	double SSE, MSE, PSNR;
	int offset;
	SSE = 0.0;

	for(int rows = 0; rows < height; rows++)
	{
		for(int cols = 0; cols < width; cols++)
		{
			offset = rows * width + cols;
			SSE += ((Ori[offset] - Result[offset]) * (Ori[offset] - Result[offset]));
		}
	}
	if(SSE == 0.0)
		return 0.0;
	else
	{
		MSE = SSE / (double)(width * height);
		PSNR = 10.0 * log10((255 * 255) / MSE);
		return PSNR;
	}
}
void parse(int argc, char* argv[]){
	//Read  CommandLine
	string Cfg_Name, Parameter[14];
	if(argc > 0)
		Cfg_Name = argv[1];
	ifstream cfg(Cfg_Name);
	if(! cfg)
	{
		cerr<<"OPEN Configure File ERROR"<<endl;
	}
	for(int i = 0; i < 14; i++)
	{
		getline(cfg,Parameter[i]);
	}
	cfg.close();

	origfile_name = Parameter[1];
	recfilename = Parameter[3];
	result_name = Parameter[5];
	stringstream ss;
	ss<<Parameter[7];
	ss>>WIDTH;
	ss.clear();
	ss<<Parameter[9];
	ss>>HEIGHT;
	ss.clear();
	ss<<Parameter[11];
	ss>>FRAME_NUMER;
	ss.clear();
	ss<<Parameter[13];
	ss>>noise_sigma;
	ss.clear();
	FRAME_SIZE = ((WIDTH * HEIGHT) * 3 / 2);
	cout<<noise_sigma<<endl;
}
int main(int argc, char* argv[])
{
	parse(argc, argv);
	//Open file
	ifstream fin_rec, fin_orig;

	fin_rec.open(recfilename,ios::in|ios::binary);
	fin_orig.open(origfile_name,ios::in|ios::binary);
	if(fin_rec.fail() || fin_orig.fail())
	{
		cout<<"The file is error!"<<endl;
		return -1;
	}
#if NLM_Filter

	ofstream fout_filtered;
	double ave_PSNR_before[3] = {0.0};
	double ave_PSNR_after[3] = {0.0};

	fout_filtered.open(result_name,ios::out|ios::binary);
	if(fout_filtered.fail())
	{
		cout<<"The file is error!"<<endl;
		return -1;
	}
#endif


	//Create YUV buffer
	char* RecY_buffer = new char[WIDTH * HEIGHT];
	char* RecCb_buffer = new char[WIDTH * HEIGHT / 4];
	char* RecCr_buffer = new char[WIDTH * HEIGHT / 4];
	char* OriY_buffer = new char[WIDTH * HEIGHT];
	char* OriCb_buffer = new char[WIDTH * HEIGHT / 4];
	char* OriCr_buffer = new char[WIDTH * HEIGHT / 4];

#if NLM_Filter
	uchar* Filtered_Y_buffer = new uchar[WIDTH * HEIGHT];
	uchar* Filtered_Cb_buffer = new uchar[WIDTH * HEIGHT / 4];
	uchar* Filtered_Cr_buffer = new uchar[WIDTH * HEIGHT / 4];
#endif

#if PrintReconstructionError
	//重建误差
	uchar* Diff_Y = new uchar[WIDTH * HEIGHT];
	uchar* Diff_Cb = new uchar[WIDTH * HEIGHT / 4];
	uchar* Diff_Cr = new uchar[WIDTH * HEIGHT / 4];

	ofstream difference;
	difference.open(result_name,ios::out | ios::binary);
	if(!difference)
	{
		cout<<"The file is error!"<<endl;
		return -1;
	}

	ofstream estimation_var;
	char* est_name = new char[40];
	sprintf(est_name, "%s%d.txt", "./Filtered/estimation_", (int)noise_sigma);
	estimation_var.open(est_name, ios::app | ios::binary);
	if(!estimation_var){
		cerr<<"estimation.txt can't open!";
	}
	memset(Diff_Cb, 128, WIDTH * HEIGHT / 4);
	memset(Diff_Cr, 128, WIDTH * HEIGHT / 4);
#endif

	for(int i = 0; i < FRAME_NUMER; i++)
	{
		fin_rec.seekg(i * FRAME_SIZE,ios::beg);
		fin_rec.read(RecY_buffer, WIDTH * HEIGHT );
		fin_rec.read(RecCb_buffer, WIDTH * HEIGHT / 4);
		fin_rec.read(RecCr_buffer, WIDTH * HEIGHT / 4);
		fin_orig.seekg(i * FRAME_SIZE, ios::beg);
		fin_orig.read(OriY_buffer, WIDTH * HEIGHT);
		fin_orig.read(OriCb_buffer, WIDTH * HEIGHT / 4);
		fin_orig.read(OriCr_buffer, WIDTH * HEIGHT / 4);

		uchar* RecY = (uchar*)RecY_buffer;
		uchar* RecCb = (uchar*)RecCb_buffer;
		uchar* RecCr = (uchar*)RecCr_buffer;
		uchar* OriY = (uchar*)OriY_buffer;
		uchar* OriCb = (uchar*)OriCb_buffer;
		uchar* OriCr = (uchar*)OriCr_buffer;

#if NLM_Filter
		memset(Filtered_Y_buffer, 0, WIDTH * HEIGHT);
		memset(Filtered_Cb_buffer, 0, WIDTH * HEIGHT / 4);
		memset(Filtered_Cr_buffer, 0, WIDTH * HEIGHT / 4);
#endif

#if PrintReconstructionError
		
		//计算均值方差，QP
		int max = 0, min = 255;
		double means = 0, vars = 0;
		for(int j = 0; j < HEIGHT; j++)
			for(int i = 0; i < WIDTH; i ++)
			{
				int diff_pel = RecY[j * WIDTH + i] - OriY[j * WIDTH + i];
				means += diff_pel;
				Diff_Y[j * WIDTH + i] = abs(diff_pel);
				if(Diff_Y[j * WIDTH + i] > max)
					max = Diff_Y[j * WIDTH + i];
				if(Diff_Y[j * WIDTH + i] < min)
					min = Diff_Y[j * WIDTH + i];
			}
		means /= (HEIGHT * WIDTH);
		//对结果拉伸
		if(max > min)
			for(int j = 0; j < HEIGHT; j++)
				for(int i = 0; i < WIDTH; i ++)
				{
					int diff_pel = RecY[j * WIDTH + i] - OriY[j * WIDTH + i];
					vars += (diff_pel - means) * (diff_pel - means);
					Diff_Y[j * WIDTH + i] = (Diff_Y[j * WIDTH + i] - min) * 255 / (max - min);

				}
		vars /= (HEIGHT * WIDTH);


		estimation_var<<"均值："<<left<<setw(20)<<means;
		estimation_var<<"方差："<<left<<setw(20)<<vars;
		estimation_var<<"标准差："<<left<<setw(20)<<sqrt(vars)<<endl;
		difference.write((char*)Diff_Y, WIDTH * HEIGHT);
		difference.write((char*)Diff_Cb, WIDTH * HEIGHT / 4);
		difference.write((char*)Diff_Cr, WIDTH * HEIGHT / 4);
#endif

#if NLM_Filter
		//Calculate PSNR before filtered
		double PSNR_before[3], PSNR_after[3];
		PSNR_before[0] = calcPSNR(OriY, RecY, WIDTH, HEIGHT);
		PSNR_before[1] = calcPSNR(OriCb, RecCb, WIDTH/2, HEIGHT/2);
		PSNR_before[2] = calcPSNR(OriCr, RecCr, WIDTH/2, HEIGHT/2);
		
		cout<<"Frame:"<<i<<endl;
		cout<<"Before_Filter(PSNR):"<<endl;
		cout<<"Y:"<<PSNR_before[0]<<"\t"<<"Cb:"<<PSNR_before[1]<<"\t"<<"Cr:"<<PSNR_before[2]<<"\t"<<endl;

		//Non-Local-Means algorithm
		Non_Local_Means(RecY, Filtered_Y_buffer, WIDTH, HEIGHT, TemplateWindowSize, SearchWindowSize, noise_sigma, noise_sigma);
		Non_Local_Means(RecCb, Filtered_Cb_buffer, WIDTH/2, HEIGHT/2, TemplateWindowSize/2, SearchWindowSize, noise_sigma, noise_sigma);
		Non_Local_Means(RecCr, Filtered_Cr_buffer, WIDTH/2, HEIGHT/2, TemplateWindowSize/2, SearchWindowSize, noise_sigma, noise_sigma);
		fout_filtered.write((char*)Filtered_Y_buffer, WIDTH*HEIGHT);
		fout_filtered.write((char*)Filtered_Cb_buffer, WIDTH*HEIGHT/4);
		fout_filtered.write((char*)Filtered_Cr_buffer, WIDTH*HEIGHT/4);

		PSNR_after[0] = calcPSNR(OriY, Filtered_Y_buffer, WIDTH, HEIGHT);
		PSNR_after[1] = calcPSNR(OriCb, Filtered_Cb_buffer, WIDTH/2, HEIGHT/2);
		PSNR_after[2] = calcPSNR(OriCr, Filtered_Cr_buffer, WIDTH/2, HEIGHT/2);

		ave_PSNR_before[0] += PSNR_before[0];
		ave_PSNR_before[1] += PSNR_before[1];
		ave_PSNR_before[2] += PSNR_before[2];
		ave_PSNR_after[0] += PSNR_after[0];
		ave_PSNR_after[1] += PSNR_after[1];
		ave_PSNR_after[2] += PSNR_after[2];

		cout<<"After_Filter(PSNR):"<<endl;
		cout<<"Y:"<<PSNR_after[0]<<"\t"<<"Cb:"<<PSNR_after[1]<<"\t"<<"Cr:"<<PSNR_after[2]<<"\t"<<endl;
		//Output filtered results as YUV file
#endif
	}

#if NLM_Filter
	cout<<"*********************"<<endl<<"AVERAGE:"<<endl;
	cout<<"ave_PSNR_before:"<<ave_PSNR_before[0]/FRAME_NUMER<<"\t"<<ave_PSNR_before[1]/FRAME_NUMER<<"\t"<<ave_PSNR_before[2]/FRAME_NUMER<<endl;
	cout<<"ave_PSNR_after:"<<ave_PSNR_after[0]/FRAME_NUMER<<"\t"<<ave_PSNR_after[1]/FRAME_NUMER<<"\t"<<ave_PSNR_after[2]/FRAME_NUMER<<endl;
	cout<<"*********************"<<endl<<"DIFFERENCE:"<<endl;
	cout<<(ave_PSNR_after[0]-ave_PSNR_before[0])/FRAME_NUMER<<"\t"<<(ave_PSNR_after[1]-ave_PSNR_before[1])/FRAME_NUMER<<"\t"<<(ave_PSNR_after[2]-ave_PSNR_before[2])/FRAME_NUMER<<endl;
#endif
	//Destroy YUV buffer
	delete[] RecY_buffer;
	delete[] RecCb_buffer;
	delete[] RecCr_buffer;
	delete[] OriY_buffer;
	delete[] OriCb_buffer;
	delete[] OriCr_buffer;

#if NLM_Filter
	delete[] Filtered_Y_buffer;
	delete[] Filtered_Cb_buffer;
	delete[] Filtered_Cr_buffer;
	fout_filtered.close();
#endif
	//Close file
	fin_rec.close();
	fin_orig.close();

#if PrintReconstructionError
	difference.close();
	estimation_var.close();
#endif

	return 0;
}
 
