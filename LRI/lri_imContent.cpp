//c++ libraries
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <bitset>
#include <stdlib.h>
//opencv libraries
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
//defines
#define LBP_NO_PIXELS 8

template<std::size_t N>
bool operator<(const std::bitset<N>& x, const std::bitset<N>& y)
{
    for (int i = N-1; i >= 0; i--) {
        if (x[i] ^ y[i]) return y[i];
    }
    return false;
}

std::vector< std::vector<float> > create_kernels(int no_directions, int k_size)
{
	std::vector< std::vector<float> > kernels;
	
	for(int d = 0; d < no_directions; d++)
	{
		//+traverse through the different kernel sizes
		for(int i = 1; i <= (k_size+1); i++)
		{
			std::vector<float> data;
			int kernel_size = (int)(std::pow( (double)(i+1), 2. ));

			switch(d)
			{
				case 0: //horizontal kernel				
					data.push_back(1.);
					for(int j = 0; j < (i-1); j++)
						data.push_back(0.);
					data.push_back(-1.);
					break;	
				
				case 1: //vertical kernel
					data.push_back(1.);
					for(int j = 0; j < (i-1); j++)
						data.push_back(0.);
					data.push_back(-1.);
					break;	
					
				case 2: // 45 downwards 
					data.push_back(1.);
					for(int j = 0; j < (kernel_size-2); j++)
						data.push_back(0.);
					data.push_back(-1.);
					break;
				
				case 3: // 45 upwards
					data.push_back(-1.);
					for(int j = 0; j < (kernel_size-2); j++)
						data.push_back(0.);
					data.push_back(1.);
					break;
				default:
					std::cout << "Invalid direction" << std::endl;
								
			}	
			kernels.push_back(data);
		}
	}
	/*
	//===DEBUGGING===//
	//print the computed kernels
	for(std::vector< std::vector<float> >::iterator it = kernels.begin(); it != kernels.end(); ++it)
	{
		std::vector<float> tmp = *it;
		for(std::vector<float>::iterator it2 = tmp.begin(); it2 != tmp.end(); ++it2)
		{
			std::cout << *it2 << " ";
		}
		std::cout << std::endl;
	}
	//===//
	*/
	
	return kernels;
}

std::vector<cv::Mat> compute_diff_im(const cv::Mat& query_im, int no_directions, int k_size, float* thresh, float thresh_alpha, 
	std::vector< std::vector<float> > kernels)
{
	std::vector<cv::Mat> diff_im_vec;
		
	//+compute the threshold value fo the image
	cv::Scalar mean, stddev;
	cv::meanStdDev(query_im, mean, stddev);
	*thresh = stddev.val[0]*thresh_alpha;
	
	
	//===DEBUGGING===//
	//print image std/threshold
	//std::cout << "mean: " << mean.val[0] << std::endl;
	//std::cout << "std: " << stddev.val[0] << std::endl;
	//===//
	
	
	//+global parameters to convolve kernels with the image
	cv::Point anchor;
	double delta = 0.;
	cv::Ptr<cv::FilterEngine> im_filter;
	
	for(int c = 0; c < 2; c++) //this is just to stack the diff_im in a more convenvient manner
	{
		for(int d = 0; d < no_directions; d++)
		{
			for(int i = 0; i < (k_size+1); i++)
			{
				//===DEBUGGING===//
				//std::cout << "* " << d << " * " << (d*(k_size+1)+i) << std::endl;
				//===//
				cv::Mat col_kernel = cv::Mat( kernels.at( (d*(k_size+1)+i) ) );
				cv::Mat diff_im = cv::Mat(query_im.size(),query_im.type());
				cv::Mat diff_im_comp = cv::Mat(query_im.size(),query_im.type());
				cv::Mat ker;
				cv::Mat tmp_query_im;
				switch(d)
				{
					//===NOTE: to convolve matrixes it is necessary to do zero padding
					//in the edges of the source image. The fcn im_filter only does
					//this in the right and bottom edges of the images. Thus, instead
					//of using fcn copyMakeBorder, images were flipped across y or x
					//then convolved and then flipped back ===//
					 
					case 0: //diff-im with horizontal kernels
						anchor = cv::Point(0,0);
						ker = col_kernel.reshape(0,1);
						if(c==0){
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(query_im, diff_im);
							diff_im_vec.push_back(diff_im);
						}else{
							cv::flip(query_im, tmp_query_im, 1);
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(tmp_query_im, diff_im_comp);
							cv::flip(diff_im_comp, diff_im_comp, 1);
							diff_im_vec.push_back(diff_im_comp);
						}
						break;
						
					case 1: //diff-im with vertical kernel
							anchor = cv::Point(0,0);
							ker = col_kernel;
							if(c==0){
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(query_im, diff_im);
							diff_im_vec.push_back(diff_im);
						}else{						
							cv::flip(query_im, tmp_query_im, 0);
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(tmp_query_im, diff_im_comp);
							cv::flip(diff_im_comp, diff_im_comp, 0);
							diff_im_vec.push_back(diff_im_comp);
						}
						break;
					
					case 2: //diff-im with 45° downwards kernel
						anchor = cv::Point(0,0);
						ker = col_kernel.reshape(0,i+2);
						if(c==0){
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(query_im, diff_im);
							diff_im_vec.push_back(diff_im);
						}else{
							cv::flip(query_im, tmp_query_im, 1);
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(tmp_query_im, diff_im_comp);
							cv::flip(diff_im_comp, diff_im_comp, 1);
							diff_im_vec.push_back(diff_im_comp);
						}
						break;
						
					case 3: //diff-im with +45 kernel
						anchor = cv::Point(i+1,i+1);
						ker = col_kernel.reshape(0,i+2);
						if(c==0){
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(query_im, diff_im);
							diff_im_vec.push_back(diff_im);
						}else{
							cv::flip(query_im, tmp_query_im, 1);
							im_filter = cv::createLinearFilter(query_im.type(), ker.type(), ker, anchor, 
								delta, cv::BORDER_CONSTANT, cv::BORDER_CONSTANT, cv::Scalar(0));
							im_filter->apply(tmp_query_im, diff_im_comp);
							cv::flip(diff_im_comp, diff_im_comp, 1);
							diff_im_vec.push_back(diff_im_comp);
						}
						break;
						
					default:
						std::cout << "Invalid direction (2)" << std::endl;
				}
				~diff_im;
				~ker;
				~col_kernel;
				~tmp_query_im;
			}
		}
	}
	
	/*
	//===DEBUGGING===//	
	//print the difference images
	for(std::vector< cv::Mat >::iterator it = diff_im_vec.begin(); it != diff_im_vec.end(); ++it)
	{
		cv::Mat tmp = *it;
		std::cout << tmp << std::endl;
		std::cout << "***" << std::endl;
	}
	//===//
	*/
	
	
	return diff_im_vec;
	
}

void compute_LRIa(int no_directions, int k_size, float thresh, std::vector<cv::Mat> diff_im_vec, cv::Mat& histograms)
{
	
	for(int d = 0; d < no_directions*2; d++)
	{
		cv::Mat hist(1,2*k_size+1, CV_32SC1, cv::Scalar(0)); //values range is from -K to K
		
		for(int pixr = (k_size+1); pixr < (diff_im_vec.at(0).rows-(k_size+1)); pixr++)
		{
			for(int pixc = (k_size+1); pixc < (diff_im_vec.at(1).cols-(k_size+1)); pixc++)
			{
				int pix_count = 0;
				bool big_or_small = true;
				bool pix_count_flag = true;
				//std::cout << "# " << std::endl;
				
				for(int sz = 0; sz < (k_size+1) && pix_count_flag == true; sz++)
				{
					//obtain pixel-dif value from the correspondant matrix
					cv::Mat diff_im = diff_im_vec.at((d*(k_size+1))+sz);
					float val = diff_im.at<float>(pixr,pixc);
					
					if(sz == 0) //for the adjacent pixel, set gradient
					{
						if(std::abs(val) < thresh)
						{   //std::cout << "+ " << k_size << std::endl;
							hist.at<int>(0,k_size) += 1 ; pix_count_flag = false;}
						else if (val < 0) //bigger
						{ big_or_small = true; pix_count++;}
						else              //smaller
						{ big_or_small = false; pix_count++;}
					}
					else //check further adjacent pixels
					{
						if(std::abs(val) < thresh) //stop condition regardless gradient
						{
							if(big_or_small)
							{   //std::cout << "+ " << k_size+std::min(pix_count,k_size) << std::endl;
								hist.at<int>(0,k_size+std::min(pix_count,k_size)) += 1;}
							else
							{   //std::cout << "+ " << k_size-std::min(pix_count,k_size) << std::endl;
								hist.at<int>(0,k_size-std::min(pix_count,k_size)) += 1;}
							
							pix_count_flag = false;

						}
						else
						{
							if( (val < 0 && big_or_small) || (val > 0 && !big_or_small))
								pix_count++;
							else
							{
								if(big_or_small)
								{   //std::cout << "+ " << k_size+std::min(pix_count,k_size) << std::endl;
								hist.at<int>(0,k_size+std::min(pix_count,k_size)) += 1;}
								else
								{   //std::cout << "+ " << k_size-std::min(pix_count,k_size) << std::endl;
								hist.at<int>(0,k_size-std::min(pix_count,k_size)) += 1;}
								
								pix_count_flag = false;
							}
							
						}
					}
				} //end of pixels comparison - go to compare the next sz-neighbor
				if(pix_count_flag) //if the size limit was reached
				{
					if(big_or_small)
					{   //std::cout << "!! " << k_size+std::min(pix_count,k_size) << std::endl;
						hist.at<int>(0,k_size+std::min(pix_count,k_size)) += 1;}
					else
					{   //std::cout << "!! " << k_size-std::min(pix_count,k_size) << std::endl;
						hist.at<int>(0,k_size-std::min(pix_count,k_size)) += 1;}
				}
				 			 
			}//-|change row and col indexes
		}//-----|then go to evaluate the next pixel
		
		//insert the resultant histogram that represents direction d
		//std::cout << hist << std::endl;
		histograms.push_back(hist);
	}
	/*
	//===DEBUGGING===//	
	//print obtained histograms
	std::cout << "Raw Histograms: " << std::endl;
	for(int r = 0; r < histograms.rows; r++)
	{
		std::cout << histograms.row(r) << std::endl;
		std::cout << "***" << std::endl;
	}
	//===//
	*/
}

void compute_LRId(int no_directions, int k_size, float thresh, std::vector<cv::Mat> diff_im_vec, cv::Mat& histograms)
{
	//std::cout << thresh << std::endl;
	
	for(int d = 0; d < no_directions*2; d++)
	{
		cv::Mat hist(1,2*k_size+1, CV_32SC1, cv::Scalar(0)); //values range is from -K to K
		
		for(int pixr = (k_size+1); pixr < (diff_im_vec.at(0).rows-(k_size+1)); pixr++)
		{
			for(int pixc = (k_size+1); pixc < (diff_im_vec.at(1).cols-(k_size+1)); pixc++)
			{
				int pix_count = 0;
				bool pix_count_flag = true;
				
				for(int sz = 0; sz < (k_size+1) && pix_count_flag == true; sz++)
				{
					//obtain pixel-dif value from the correspondant matrix
					cv::Mat diff_im = diff_im_vec.at((d*(k_size+1))+sz);
					float val = diff_im.at<float>(pixr,pixc);
					
					if(std::abs(val) >= thresh) //if the current pixel is smaller or greater by at least T
					{
						int hist_pos = std::min(pix_count+1,k_size)%k_size;
						
						if(val < 0) //the neighbor is greater
						{	hist.at<int>(0,k_size+hist_pos) += 1; }
						else
						{	hist.at<int>(0,k_size-hist_pos) += 1; }
						
						pix_count_flag = false;
					}
					else
					{	pix_count++; }
				}
				if(pix_count_flag) //if the size limit was reached
				{	hist.at<int>(0,k_size) += 1; }
								
			}//|-- evaluate the next pixel
		}//|||||-- evaluate the next pixel
		
		//insert the resultant histogram that represents direction d
		//std::cout << hist << std::endl;
		histograms.push_back(hist);
		
	} //compute next direction histogram
	
	/*
	//===DEBUGGING===//	
	//print obtained histograms
	std::cout << "Raw Histograms: " << std::endl;
	for(int r = 0; r < histograms.rows; r++)
	{
		std::cout << histograms.row(r) << std::endl;
		std::cout << "***" << std::endl;
	}
	//===//
	*/
	
	
}

void compute_LBPbasic(int no_directions, int k_size, std::vector<cv::Mat> diff_im_vec, cv::Mat& histograms)
{
	cv::Mat hist(1, pow(2,(double)LBP_NO_PIXELS), CV_32SC1, cv::Scalar(0)); //values can be 2^8 patterns
	
	for(int pixr = (k_size+1); pixr < (diff_im_vec.at(0).rows-(k_size+1)); pixr++)
	{
		for(int pixc = (k_size+1); pixc < (diff_im_vec.at(1).cols-(k_size+1)); pixc++)
		{
			std::bitset<LBP_NO_PIXELS> code;
			
			for(int d = 0; d < no_directions*2; d++)
			{
				//obtain pixel-dif value from the correspondant matrix
				cv::Mat diff_im = diff_im_vec.at((d*(k_size+1)));
				float val = diff_im.at<float>(pixr,pixc);
				int shift_pos = 0;
								
				//===NOTE: Normally, using the value of d this can be done w/o the case
				//instruction, but because the kernels where not built as in the LBP order
				//a case statement is necessary to generate the LBP codes according to the
				//convention; AGAIN, this is not necessary for the code to work ===//
				switch(d)
				{
					case 0:
						shift_pos = 0;
						break;
					case 1:
						shift_pos = 2;
						break;
					case 2: 
						shift_pos = 1;
					case 3:
						shift_pos = 5;
						break;
					case 4:
						shift_pos = 4;
						break;
					case 5:
						shift_pos = 6;
						break;
					case 6:
						shift_pos = 3;
						break;
					case 7:
						shift_pos = 7;
						break;
					default:
						shift_pos = 0;
				}
				
				if(val <= 0){ code.set(shift_pos,1); }				
			} //go an look to the next neighbor of the target pixel	
			
			//shift the code generated to the least value possible,
			//to make LBP rotation invariant
			
			//std::cout << code << std::endl;
			
			std::bitset<1> tmp_bit;
			std::bitset<LBP_NO_PIXELS> min_code = code;
			int bit_shift_cc = 0;
			do
			{
				tmp_bit[0] = code[LBP_NO_PIXELS-1];
				code = code << 1;
				code[0] = tmp_bit[0];
				//std::cout << code << std::endl;
				if(code < min_code)
					min_code = code;
				bit_shift_cc++;
			}while(bit_shift_cc < LBP_NO_PIXELS-1);
			
			hist.at<int>( 0,(int)(min_code.to_ulong()) ) += 1;
			
			//===DEBUGGING===//
			//std::cout << min_code << std::endl;
			//std::cout << "***" << std::endl;
			//===//
			
		}//---|
	}//-------|change pixel indexes
	
	histograms.push_back(hist);
	//std::cout << histograms.size() << std::endl;
	//===DEBUGGING===//
	//std::cout << hist.row(0) << std::endl;
	//===//
	
	
}

double compute_intensity_pen(const cv::Mat& base_mat, const cv::Mat& query_mat)
{
	double penalization = 0.;
	double tmin = 10.; //intensity difference minimum
	double pl = 2; 	//severity of the penalization; the greater the value
					//the greater the severity of the penalization
	double val_range = 256; //256 because we use gray-scale images
	
	//compute the mean value for each image
	cv::Scalar base_mean, query_mean;
	base_mean = cv::mean(base_mat);
	query_mean = cv::mean(query_mat);
	
	
	double mean_dif = std::abs(base_mean.val[0]-query_mean.val[0]);
	//the next formula applies (1/1-x^2) in order to increase the value
	//as the differnce in intensity decreases
	penalization = 1/(1-pow((std::max(tmin,mean_dif)/val_range), pl));
	
	//===DEBUGGING===//
	//std::cout << base_mean << " " << query_mean << " " << mean_dif << std::endl;
	//std::cout << penalization << std::endl;
	//===//
	
	return penalization;	
	
}

void compute_norm_feat_vec(const cv::Mat& mul_feat, cv::Mat& feat_vec)
{
	feat_vec = mul_feat.reshape(0,1);
	feat_vec.convertTo(feat_vec, CV_32FC1);
	normalize(feat_vec, feat_vec, 1, 0, cv::NORM_L1, -1, cv::Mat() );
	
	//===DEBUGGING===//	
	//print feature vectors
	//std::cout << feat_vec << std::endl;
	//===//	
}

std::vector<cv::Mat> extract_patches(const cv::Mat& test_im, int patchesx, int patchesy)
{
	std::vector<cv::Mat> patches;
	
	//++SELECT THE PATCHES FROM THE IMAGE++//
	for(int i = 0; i < patchesy; i++)
	{
		for(int j = 0; j < patchesx; j++)
		{
			int col_width = test_im.cols/patchesx;
			int row_width = test_im.rows/patchesy;
			int col_pos = j*col_width;
			int row_pos = i*row_width;

			col_width = ( (test_im.cols-col_pos)<(2*col_width) ? (test_im.cols-col_pos):col_width );
			row_width = ( (test_im.cols-row_pos)<(2*row_width) ? (test_im.rows-row_pos):row_width );
			
			//===DEBUGGING===//
			//print ROI parameters
			//std::cout << "* " << col_pos << " " << col_width << " " << row_pos << " " << row_width;
			//std::cout << std::endl;
			//===//
			
			cv::Rect roi = cv::Rect(col_pos, row_pos, col_width, row_width);
			cv::Mat tmp_patch = test_im(roi);
			patches.push_back(tmp_patch);
		}
	}
	
	return patches;
}

double text_global_impurity(const cv::Mat& simMat)
{
	double imp = 0.;
	double norm_fac = 0.;
	
	for(int row = 1; row < simMat.rows; row++)
	{
		for(int col = 0; col < row; col++)
		{			
			imp += (double)simMat.at<float>(row,col);
			norm_fac = 1.;
			
			//imp += pow( (double)simMat.at<float>(row,col), 2.);
			//norm_fac += (double)simMat.at<float>(row,col);
		}
	}
	
	imp = imp / norm_fac;
	
	return imp;
		
}

double text_region_impurity(const cv::Mat& simMat, int patchesx, int patchesy)
{
	double imp = 0.;
	int coor_patch1[2], coor_patch2[2];
	double inv_total_dist = 0.;
	double norm_fac = 0.;
	double max_patch_dist = sqrt( pow((double)(patchesx-1),2.) + pow((double)(patchesy-1),2.) ) ;
	double exp_c = 3.;
	//std::cout << max_patch_dist << std::endl;
	
	
	for(int row = 1; row < simMat.rows; row++)
	{
		for(int col = 0; col < row; col++)
		{
			coor_patch1[0] = row/patchesx; //row divided by no of patches
			coor_patch1[1] = row%patchesx; //mod with no patches
			coor_patch2[0] = col/patchesx; //row divided by no of patches
			coor_patch2[1] = col%patchesx; //mod with no patches
			
			//std::cout << coor_patch1[0] << " " << coor_patch1[1] << " "
			//	<< coor_patch2[0] << " " << coor_patch2[1] << " " << std::endl;
			
			double dist = pow((double)(coor_patch1[0]-coor_patch2[0]),2.);
			dist += pow((double)(coor_patch1[1]-coor_patch2[1]),2.);
			//dist = sqrt(dist) / max_patch_dist;	
			
			//gaussian damping
			dist = sqrt(dist);
			dist = (-exp_c/max_patch_dist)*(dist-1.);
			dist = exp(dist);		
			//std::cout << dist << std::endl;
			
			//imp = imp + ((double)simMat.at<float>(row,col)/dist);
			//norm_fac += (1/dist);
			
			//gaussian damping
			imp = imp + ((double)simMat.at<float>(row,col)*dist);
			norm_fac += dist;
			
			//imp = imp + ( pow( (double)simMat.at<float>(row,col), 2.) / dist );
			//norm_fac += ((double)simMat.at<float>(row,col) / dist);
		}
	}
	
	imp = imp / norm_fac;
	
	return imp;
}

float convert_entropyToVel(double entropy, int metric_mode)
{
	float velocity = 0.;
	
	//===NOTE: The idea is to map the entropy values to a value between 0 and 1
	//Depending on the chosen metric, entropy values vary for homogeneous
	//and non-homogeneous textures. Based on tests, the following parameters
	//are chosen ===//
	
	/*
	//Inverse sigmoid function------
	float inflection_point = 0.;
	float offset = 0.5;
	float normc = 1.;
	float eps_c = 0.005;
	
	if(metric_mode==0)
	{	inflection_point = 0.065*2;}
	else
	{	inflection_point = 0.105*2;}
	
	//compute normalization factor to keep values between 0 and 1
	normc = inflection_point / eps_c;
	normc = 2*log(normc); //to make the maximum value 0.5
	
	//compute velocity, check logarithm case
	velocity = entropy/(inflection_point-entropy);
	if(velocity <= 0)
		velocity = 1.;
	else
		velocity = 1-((log(velocity)/normc)+offset);
		
	*/
	
	//Linear interpolation
	float max_ent, min_ent;

	if(metric_mode == 0)
	{
		min_ent = 0.018;
		max_ent = 0.115;
	}
	else if(metric_mode == 1)
	{
		min_ent = 0.024;
		max_ent = 0.175;
	}
	else
	{
		min_ent = 0.05;
		max_ent = 0.135;		
	}
	
	velocity = ( (.99/(min_ent-max_ent))*entropy )+ (1.-( (.99*min_ent)/(min_ent-max_ent) ) ) ;
	
	//bound velocity
	if(velocity < 0)
		velocity = 0.;
	else if(velocity > 1)
		velocity = 1.;
	
	
	return velocity;
}

void compute_simMat(const cv::Mat& hist_vec, cv::Mat& simMat, bool flag_intensity_pen, std::vector<cv::Mat> patches)
{
	//++HISTOGRAM COMPARISON++//
	//+CREATE SIMILARITY MATRIX BETWEEN THE PATCHES+//
	//===NOTE: only the lower triangle sub-matrix of this matrix contains values
	//this is done to make the element-access and traversal easier===//
	//cv::Mat simMat((int)hist_vec.size(), (int)hist_vec.size(), CV_32FC1, cv::Scalar(0.));
	
	for(int sim_row = 1; sim_row < simMat.rows; sim_row++)
	{
		for(int sim_col = 0; sim_col < sim_row; sim_col++)
		{
			double score = compareHist(hist_vec.row(sim_row), hist_vec.row(sim_col), CV_COMP_BHATTACHARYYA);
			if(flag_intensity_pen)
				simMat.at<float>(sim_row,sim_col) = score*compute_intensity_pen(patches.at(sim_row),patches.at(sim_col));
			else
				simMat.at<float>(sim_row,sim_col) = score;
			//std::cout << score << std::endl;
		}
	}
	
	/*
	//===DEBUGGING===//
	//print simmilarity matrix
	std::cout << "Simmilarity matrix" << std::endl;
	for(int row = 0; row < simMat.rows; row++)
	{
		std::cout << simMat.row(row) << std::endl;
	}
	//===//
	*/
		
}

void compute_combined_simMat(const cv::Mat& hist_vec1, const cv::Mat& hist_vec2, double w1, double w2, 
	cv::Mat& final_simMat, bool flag_intensity_pen, std::vector<cv::Mat> patches)
{
	cv::Mat simMat1((int)hist_vec1.rows, (int)hist_vec1.rows, CV_32FC1, cv::Scalar(0.));
	cv::Mat simMat2((int)hist_vec2.rows, (int)hist_vec2.rows, CV_32FC1, cv::Scalar(0.));
	
	compute_simMat(hist_vec1, simMat1, flag_intensity_pen, patches);		
	compute_simMat(hist_vec2, simMat2, flag_intensity_pen, patches);	
	
	for(int sim_row = 1; sim_row < final_simMat.rows; sim_row++)
	{
		for(int sim_col = 0; sim_col < sim_row; sim_col++)
		{
			double score = pow( simMat1.at<float>(sim_row, sim_col), w1 ) * pow( simMat2.at<float>(sim_row, sim_col), w2 );
			if(flag_intensity_pen)
				score *= compute_intensity_pen(patches.at(sim_row),patches.at(sim_col));
			
			final_simMat.at<float>(sim_row, sim_col) = score;

			//std::cout << score << std::endl;
		}
	}
	
	/*
	//===DEBUGGING===//
	//print simmilarity matrix
	std::cout << "Simmilarity matrix" << std::endl;
	for(int row = 0; row < final_simMat.rows; row++)
	{
		std::cout << final_simMat.row(row) << std::endl;
	}
	//===//
	*/


}

double compute_text_entropy(const cv::Mat& test_im, int no_directions, int k_size, float* thresh, float thresh_alpha, 
	std::vector< std::vector<float> > kernels, int patchesx, int patchesy, int mode, bool flag_intensity_pen)
{
	double text_entropy = 0.;
	
	//++EXTRACT PATCHES FROM THE IMAGE++//
	std::vector<cv::Mat> patches(extract_patches(test_im, patchesx, patchesy));
	
	//++COMPUTE DIFFERENCE IMAGES AND HISTOGRAMS++//
	//std::vector<cv::Mat> hist_vec;
	cv::Mat hist_vec_lri, hist_vec_lbp;
	//std::cout << hist_vec_lri.empty() << " " << hist_vec_lbp.empty() << std::endl;
	
	for(std::vector<cv::Mat>::size_type i = 0; i != patches.size(); i++)
	{
		cv::Mat tmp_hist_lri(0,2*k_size+1, CV_32SC1);
		cv::Mat tmp_hist_lbp(0, pow(2,(double)LBP_NO_PIXELS), CV_32SC1); //values can be 2^8 patterns
		cv::Mat tmp_featvec;
		std::vector<cv::Mat> difference_images_patch( compute_diff_im(patches[i], no_directions, k_size, thresh, thresh_alpha, kernels) );
		
		//Compute LRIa
		if(mode == 0)
		{
			compute_LRIa(no_directions, k_size, *thresh, difference_images_patch, tmp_hist_lri);
		}
		else if(mode == 1)
		{
			compute_LRId(no_directions, k_size, *thresh, difference_images_patch, tmp_hist_lri);
		}
		else if(mode == 2)
		{
			compute_LBPbasic(no_directions, k_size, difference_images_patch, tmp_hist_lbp);
		}
		else if(mode == 3)
		{ 
			compute_LRIa(no_directions, k_size, *thresh, difference_images_patch, tmp_hist_lri);
			compute_LBPbasic(no_directions, k_size, difference_images_patch, tmp_hist_lbp);
		}
		else
		{
			compute_LRId(no_directions, k_size, *thresh, difference_images_patch, tmp_hist_lri);
			compute_LBPbasic(no_directions, k_size, difference_images_patch, tmp_hist_lbp);
		}
		
		if(mode == 0 || mode == 1)
		{
			compute_norm_feat_vec(tmp_hist_lri, tmp_featvec);
			hist_vec_lri.push_back(tmp_featvec);
		}
		else if(mode == 2)
		{
			compute_norm_feat_vec(tmp_hist_lbp, tmp_featvec);
			hist_vec_lbp.push_back(tmp_featvec);
		}	
		else
		{
			compute_norm_feat_vec(tmp_hist_lri, tmp_featvec);
			hist_vec_lri.push_back(tmp_featvec);
			compute_norm_feat_vec(tmp_hist_lbp, tmp_featvec);
			hist_vec_lbp.push_back(tmp_featvec);
		}
		
	}
	
	//std::cout << hist_vec_lri.empty() << " " << hist_vec_lbp.empty() << std::endl;
	
	//cv::Mat simMat((int)hist_vec.size(), (int)hist_vec.size(), CV_32FC1, cv::Scalar(0.));
	double w1 = 1.;
	double w2 = 1.;
	cv::Mat hist_vec;
	int simMat_size = patchesx*patchesy;
	cv::Mat simMat(simMat_size, simMat_size, CV_32FC1, cv::Scalar(0.));
	if(mode == 0 || mode == 1)
	{
		compute_simMat(hist_vec_lri, simMat, flag_intensity_pen, patches);		
	}
	else if(mode == 2)
	{
		compute_simMat(hist_vec_lbp, simMat, flag_intensity_pen, patches);		
	}
	else
	{
		compute_combined_simMat(hist_vec_lri, hist_vec_lbp, w1, w2, simMat, flag_intensity_pen, patches);
	}	

	/*
	//++HISTOGRAM COMPARISON++//
	//+CREATE SIMILARITY MATRIX BETWEEN THE PATCHES+//
	//===NOTE: only the lower triangle sub-matrix of this matrix contains values
	//this is done to make the element-access and traversal easier===//
	cv::Mat simMat((int)hist_vec.size(), (int)hist_vec.size(), CV_32FC1, cv::Scalar(0.));
	
	for(int row = 1; row < simMat.rows; row++)
	{
		for(int col = 0; col < row; col++)
		{
			double score = compareHist(hist_vec.at(row), hist_vec.at(col), CV_COMP_BHATTACHARYYA);
			if(flag_intensity_pen)
				simMat.at<float>(row,col) = score*compute_intensity_pen(patches.at(row),patches.at(col));
			else
				simMat.at<float>(row,col) = score;
			//std::cout << score << std::endl;
		}
	}
	*/
	
	/*
	//===DEBUGGING===//
	//print simmilarity matrix
	std::cout << "Simmilarity matrix" << std::endl;
	for(int row = 0; row < simMat.rows; row++)
	{
		std::cout << simMat.row(row) << std::endl;
	}
	//===//
	*/
	

	//+++COMPUTE GLOBAL IMPURITY+++//
	double global_impurity = text_global_impurity(simMat);
	//std::cout << "Global impurity: " << global_impurity << std::endl;
	
	//+++COMPUTE REGION-BASED IMPURITY+++//
	double weighted_impurity = text_region_impurity(simMat, patchesx, patchesy);
	//std::cout << "Weighted impurity: " << weighted_impurity << std::endl;
	//std::cout << "Log e: " << log(weighted_impurity) << std::endl;
	//std::cout << "Log 10: " << log10(weighted_impurity) << std::endl;
	std::cout << weighted_impurity << std::endl;
	//std::cout << log(weighted_impurity) << std::endl;

	
	//float vel = convert_entropyToVel(weighted_impurity, mode);
	//std::cout << "Vel: " << vel << std::endl;

	
	/*
	//+++MAP TO VELOCITIES++//
	double cp = 0.5;
	double tp = 12;
	double vel = cp*tan(tp*weighted_impurity);
	std::cout << "Velocity: " << vel << std::endl;
	*/
	
	
	
	return text_entropy;
}

//====================================================================================================================================
//*** MAIN METHOD ***//
int main ( int argc, char *argv[] )
{
	//+++INITIAL INPUT PARAMETERS+++//
	//1) source image 2) neighborhood size 3) threshold
	std::string test_im_name = argv[1];
	cv::Mat test_im = cv::imread(test_im_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	test_im.convertTo(test_im, CV_32FC1);
	
	/*
	//just for testing
	cv::Mat query_im = cv::imread("/home/arturokkboss33/DataSets/Gustaf/floor1/floor1-a-p003.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Rect roi = cv::Rect(0,0,10,10);
	cv::Mat test_im = query_im(roi);
	test_im.convertTo(test_im, CV_32FC1);
	*/
	
	int k_size = atoi(argv[2]);
	float thresh = 0.;
	float thresh_alpha = atof(argv[3]);
	int no_patchesx = atoi(argv[4]);
	int no_patchesy = atoi(argv[5]);
	int no_directions = 4;
	int mode = 4;
	bool penalty = false;
	
	//+++CREATE KERNELS FOR DIFFERENCE IMAGES+++//
	std::vector< std::vector<float> > kernels( create_kernels(no_directions, k_size) );
	
	//+++DIVIDE THE IMAGE INTO PATCHES AND COMPUTE THE INTRA-SIMMILARITY+++//
	compute_text_entropy(test_im, no_directions, k_size, &thresh, thresh_alpha, kernels, no_patchesx, no_patchesy, mode, penalty);
	
	
	//+++
	
	/*
	//+++DIFFERENCE IMAGE AND HISTOGRAM COMPUTATION+++//
	cv::Mat histograms_query(0,2*k_size+1, CV_32SC1);
	cv::Mat histograms_base(0,2*k_size+1, CV_32SC1);
	std::vector<cv::Mat> difference_images_query( compute_diff_im(test_im, no_directions, k_size, &thresh, thresh_alpha, kernels) );
	compute_LRIa(no_directions, k_size, thresh, difference_images_query, histograms_query);
	std::vector<cv::Mat> difference_images_base( compute_diff_im(base_im, no_directions, k_size, &thresh, thresh_alpha, kernels) );
	compute_LRIa(no_directions, k_size, thresh, difference_images_base, histograms_base);
	
	//+++HISTOGRAM COMPARISON+++
	cv::Mat feature_query, feature_base;
	compute_norm_feat_vec(histograms_query, feature_query);
	compute_norm_feat_vec(histograms_base, feature_base);
	
	//===NOTE: The similiraity score using BHATACHARYYA ranges from 0 to 1
	//where 0 indicates that two images are the same ===//
	double score = compareHist(feature_query, feature_base, CV_COMP_BHATTACHARYYA);
	std::cout << "Similarity score: " << score << std::endl;
	*/
	

	return 0;
}
