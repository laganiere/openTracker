
#if !defined VTRACKER
#define VTRACKER

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "videoprocessor.h"

// The tracker class 
class VisualTracker : public FrameProcessor {

	  // feature detector and descriptor
	  cv::Ptr<cv::FeatureDetector> detector;
	  cv::Ptr<cv::DescriptorExtractor> descriptor;
	  // feature matcher
	  cv::BFMatcher matcher;

	  // the current bounding box
	  cv::Rect bbox;

	  // target model
	  cv::Mat targetImage;
	  std::vector<cv::KeyPoint> targetPoints;
	  cv::Mat targetDescriptors;
	  // radius for context
	  int radius;
	  // descriptors in context
	  std::vector<cv::Mat> context;

	  // search area
	  cv::Rect searchArea;

	  // matches
	  std::vector<cv::DMatch> targetMatches;

  public:

	  VisualTracker() : matcher(cv::NORM_L2) {

		  // Construct the SIFT feature detector object
		  detector = new cv::SIFT();
		  // Construct the SIFT feature descriptor object
		  descriptor = new cv::SIFT();

		  // initial bounding box
		  searchArea.width = 120;
		  searchArea.height = 90;
		  // outer context
		  radius = 20;
	  }

	  // create context and target model
	  void createTarget(cv::Mat &image, cv::Rect &initialBB) {

		  bbox = initialBB;
		  cv::Mat(image,initialBB).copyTo(targetImage);

		  // defining the ROI
		  cv::Rect outBbox = cv::Rect(bbox.x - radius, bbox.y - radius, bbox.width + 2*radius, bbox.height + 2*radius);
		  cv::Mat roi(image, outBbox);

		  // Detect the SIFT features
		  std::vector<cv::KeyPoint> keypoints;
		  detector->detect(roi, keypoints);

		  // the inner bounding box bottom right corner
		  int xf = radius + bbox.width;
		  int yf = radius + bbox.height;

		  // context features 
		  std::vector<cv::KeyPoint> contextPoints;

		  // moving the keypoints
		  for (cv::KeyPoint &point : keypoints) {

			  // is it outside the target BB?
			  if (point.pt.x<radius || point.pt.y<radius || point.pt.x>xf || point.pt.y>yf) {

				  contextPoints.push_back(point);
			  }
			  else { // it is a model point

				  targetPoints.push_back(point);
			  }
		  }

		  // Describe the SIFT features
		  cv::Mat descriptors;
		  descriptor->compute(roi, contextPoints, descriptors);
		  context.push_back(descriptors);
		  descriptor->compute(roi, targetPoints, targetDescriptors);

		  // translate feature points (for display only)
		  for (cv::KeyPoint &point : contextPoints) {
			  point.pt.x += outBbox.x;
			  point.pt.y += outBbox.y;
		  }
		  for (cv::KeyPoint &point : targetPoints) {
			  point.pt.x -= radius;
			  point.pt.y -= radius;
		  }


		  // draw the keypoints
		  cv::drawKeypoints(image, contextPoints, image);
		  cv::rectangle(image, bbox, cv::Scalar(255, 255, 255), 3);
		  cv::imshow("Context", image);
		  cv::drawKeypoints(targetImage, targetPoints, targetImage);
		  cv::imshow("Target", targetImage);

		  std::vector<cv::DMatch> matches;
	  }

	  // processing method
	  void process(cv::Mat &input, cv::Mat &output) {

		  // defining the ROI
		  searchArea.x = bbox.x + bbox.width / 2 - searchArea.width / 2;
		  searchArea.y = bbox.y + bbox.height / 2 - searchArea.height / 2;
		  cv::Mat roi(input, searchArea);

		  // Detect the SIFT features
		  std::vector<cv::KeyPoint> keypoints;
		  detector->detect(roi, keypoints);

		  // Describe the SIFT features
		  cv::Mat descriptors;
		  descriptor->compute(roi, keypoints, descriptors);

		  // translate feature points (for display only)
		  for (cv::KeyPoint &point : keypoints) {
			  point.pt.x += searchArea.x;
			  point.pt.y += searchArea.y;
		  }

		  // draw the keypoints
		  cv::drawKeypoints(input, keypoints, output);

		  // matching phase
		  std::vector<std::vector<cv::DMatch>> matches;
		  double ratio = 0.85;
		  std::vector<std::vector<cv::DMatch>>::iterator it;

		  // context matching
		  // find the best two matches of each keypoint
		  matcher.knnMatch(descriptors, context[0], matches,
			  2); // find the k best matches

		  // masking of the context matches
		  cv::Mat mask(cv::Size(targetDescriptors.rows, descriptors.rows), CV_8U, 1);
		  cv::Mat row(cv::Size(targetDescriptors.rows,1), CV_8U, 0);

		  // perform ratio test
		  for (it = matches.begin(); it != matches.end(); ++it) {

			  //   first best match/second best match
			  if ((*it)[0].distance / (*it)[1].distance < ratio) {
				  // it is an acceptable match
				  targetMatches.push_back((*it)[0]);
			  }
		  }

		  for (int i = 0; i < mask.rows; i++) {
//			  row.copyTo(mask.row(i));
			  std::cout << row.rows << " et " << mask.row(i).rows << std::endl;
			  mask.at<uchar>(0, i)=0;
		  }

		  // target matching
		  // find the best two matches of each keypoint
		  matches.clear();
		  matcher.knnMatch(descriptors, targetDescriptors, matches,
			  2, // find the k best matches
			  mask); // do not match decriptors matching with context
		  targetMatches.clear();

		  // perform ratio test
		  for (it = matches.begin(); it != matches.end(); ++it) {

			  //   first best match/second best match
			  if ((*it)[0].distance / (*it)[1].distance < ratio) {
				  // it is an acceptable match
				  targetMatches.push_back((*it)[0]);
			  }
		  }


		  // draw matches
		  cv::Mat imageMatches;
		  cv::drawMatches(
			  input, keypoints, // 1st image and its keypoints
			  targetImage, targetPoints, // 2nd image and its keypoints
			  targetMatches,            // the matches
			  imageMatches,      // the image produced
			  cv::Scalar(255, 0, 0),  // color of lines
			  cv::Scalar(255, 255, 255)); // color of points

		  // Display the image of matches
		  cv::imshow("Matches", imageMatches);

	  }
};

#endif
