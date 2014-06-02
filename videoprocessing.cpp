/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 11 of the cookbook:  
   Computer Vision Programming using the OpenCV Library 
   Second Edition 
   by Robert Laganiere, Packt Publishing, 2013.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "videoprocessor.h"
#include "visualTracker.h"

int main()
{
	// Now using the VideoProcessor class

	// Create instance
	VideoProcessor processor;
	cv::Ptr<VisualTracker> framer = new VisualTracker;

	// Open video file
	processor.setInput("bike.avi");

	// Declare a window to display the video
	processor.displayInput("Input Video");
	processor.displayOutput("Output Video");

	// Play the video at the original frame rate
	processor.setDelay(5);// 1000. / processor.getFrameRate());

	// Set the frame processor callback function
	processor.setFrameProcessor(framer);

	// Get first frame
	cv::Mat initialFrame;
	processor.readNextFrame(initialFrame);

	// Get initial bounding box
	cv::Rect bbox;
	bbox.x = 75;
	bbox.y = 60;
	bbox.width= 120;
	bbox.height = 90;

	// create initial target model
	framer->createTarget(initialFrame, bbox);

	// Start the process
	processor.run();

	cv::waitKey();	
}