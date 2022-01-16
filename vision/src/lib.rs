use opencv::prelude::*;
use opencv::core::{Scalar, CV_16S, CV_8UC1};
use opencv::highgui::{named_window, imshow, wait_key, WINDOW_AUTOSIZE, WINDOW_NORMAL};
use opencv::imgproc::{cvt_color, COLOR_RGB2GRAY};
use opencv::calib3d::{StereoBM, StereoMatcher};
use opencv::videoio::{VideoCapture, VideoCaptureTrait};

pub struct VSLAM;

impl VSLAM {
  pub fn new() -> Self {
    Self {}
  }

  pub async fn run(&self) -> Result<(), String> {
    println!("vision");
    const WIDTH: i32 = 1920;
    const HEIGHT: i32 = 1080;
  
    opencv::core::set_log_level(opencv::core::LogLevel::LOG_LEVEL_INFO).unwrap();
  
    let mut left = VideoCapture::from_file("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", opencv::videoio::CAP_GSTREAMER).unwrap();
    let mut right = VideoCapture::from_file("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", opencv::videoio::CAP_GSTREAMER).unwrap();
    
    let mut left_frame = Mat::default();
    let mut right_frame = Mat::default();
  
    let mut left_image = Mat::default();
    let mut right_image = Mat::default();
  
    let mut disparity_16s = Mat::new_rows_cols_with_default(HEIGHT, WIDTH, CV_16S, Scalar::from(0.0)).unwrap();
    let mut disparity_8u = Mat::new_rows_cols_with_default(HEIGHT, WIDTH, CV_8UC1, Scalar::from(0.0)).unwrap();
  
    named_window("Left_Camera", WINDOW_AUTOSIZE).unwrap();
    named_window("Right_Camera", WINDOW_AUTOSIZE).unwrap();
    named_window("Disparity", WINDOW_NORMAL).unwrap();
  
    loop {
      assert!(left.read(&mut left_frame).unwrap());
      assert!(right.read(&mut right_frame).unwrap());
  
      cvt_color(&left_frame, &mut left_image, COLOR_RGB2GRAY, 0).unwrap();
      cvt_color(&right_frame, &mut right_image, COLOR_RGB2GRAY, 0).unwrap();
  
      // Range of disparity
      let n_disparities = 16 * 5;
      // Size of the block window -> must be odd
      let block_size = 21;
      let mut bm = <dyn StereoBM>::create(n_disparities, block_size).unwrap();
      // Calculate the disparity image
      bm.compute(&left_image, &right_image, &mut disparity_16s).unwrap();
      // Check its extreme values
      let mut min = 0.0;
      let mut max = 0.0;
      opencv::core::min_max_loc(&disparity_16s, Some(&mut min), Some(&mut max), None, None, &opencv::core::no_array()).unwrap();
      println!("Min disp: {} Max value: {}", min, max);
  
      imshow("Left_Camera", &left_image).unwrap();
      imshow("Right_Camera", &right_image).unwrap();
  
      disparity_16s.convert_to(&mut disparity_8u, CV_8UC1, 255.0 / (max - min), 0.0).unwrap();
  
      imshow("Disparity", &disparity_8u).unwrap();
      if wait_key(30).unwrap() == 27 {
        break;
      }
    }
    Ok(())
  }
}
