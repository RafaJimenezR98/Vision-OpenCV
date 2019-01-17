#include <opencv2/core/core.hpp> //core routines
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include "opencv2/imgproc/imgproc.hpp"//Para ecualHist
#include <iostream>

#include <exception>
#include <opencv2/core/utility.hpp>

#include "opencv2/opencv.hpp"

//#include "videoio.hpp"

#include <vector>
#include <string>
#include <sstream>

	//"-r=0.5  | valor-por-defecto |  description"
	//Cuando se usa @, se indica el orden(como un vector)
const cv::String keys =    
    "{t         		  |10                 | umbral deteccion     }"
    "{s                   |0                  | radio del elemento estructural   }"
    "{@videoInput         |video2.avi         | <input-pathname|int>   }"
    "{@videoOutput        |out.avi            | <output-pathname>   }"
    ;
    
cv::Mat frameDiference(cv::Mat frame1, cv::Mat frame2, float t, int s){
	
	cv::Mat resta;
	
	cv::absdiff(frame1,frame2,resta);
	resta = resta > t;
	
	
	if( s==0 ){ //No se hacen las operaciones estructurales	
		return resta;
	}
	
	cv::Mat outputAux;
	cv::Mat output;
	
	
	//MORPH_RECT == 0
	//cv::Mat element = cv:getStructuringElement( MORPH_RECT, Size( 2*s + 1, 2*s+1 ), Point( s, s ) );
	
	//MORPH_ELLIPSE == 2
	cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*s + 1, 2*s+1 ), cv::Point( s, s ) );
	
	//MORPH_OPEN == 2
	cv::morphologyEx( resta, outputAux, 2, element ); //Para hacer opening
	
	//MORPH_CLOSE == 3
	cv::morphologyEx( outputAux, output, 3, element ); //Para hacer closing
	
	//output=frame1 & output; //Para optimizar visualizacion
	
	//cv::Mat seg[3];
	//split(output,seg);
	
	//output=seg[0] | seg[1] | seg[2];
	
	return output;

};

cv::Mat splitOR(cv::Mat frame){
	cv::Mat seg[3];
	split(frame,seg);
	
	frame=seg[0] | seg[1] | seg[2];
	
	return frame;

};

int main(int argc,char **argv){

try{
  //augReal rows cols size intrinsics.yml <input video-file|cam-idx>
  
  if(argc<3) {std::cerr<<"USE--> ./vseg [-t <float>] [-s <int>] <input-pathname|int> <output-pathname>"<<std::endl;return 0;}   
  
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Application name v1.0.0");

	
	float t = parser.get<float>("t");
	int s = parser.get<int>("s");
	if(t <= 0){
		t=10;
	}
	if(s < 0){
		s=0;
	}
	cv::String video = parser.get<cv::String>(0);
	cv::String videoName = parser.get<cv::String>(1);
	if(videoName.empty()){
		videoName= "out.avi";
	}
	
	
	cv::VideoCapture capture(video);
	if(!capture.isOpened()){ //Error abriendo video
		std::cout<<"Error abriendo el video"<<std::endl;
		return 0;
	}
	
	int cont=0;
	int key=0;
	std::string outputName;
	
	////////For saving video Output
	double fps=25.0;
	fps= capture.get(CV_CAP_PROP_FPS);
	int frame_width=   capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height=   capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//cv::VideoWriter outputVideo(videoName,CV_FOURCC('M','J','P','G'),fps, cv::Size(frame_width,frame_height),true); //Para guardar video en color
	
	cv::VideoWriter outputVideo(videoName,0,fps, cv::Size(frame_width,frame_height),false); //Para guardar video en blanco/negro
	////////
	
	int umbral=t;
	cv::namedWindow("Display window");
	cv::createTrackbar("Umbral", "Display window", &umbral, 255);
	
	while(capture.grab()){
		cv::Mat frame1, frame2;
		capture.retrieve(frame1); //Guarda en frame el frame
		
		if((cont!=0)&&(capture.grab())){ //No es el primer frame ni el ultimo
			capture.retrieve(frame2);
			
			cv::Mat outputFrame;
			
			t=umbral;
			
			outputFrame=frameDiference(frame1, frame2, t, s);
			
			key=cv::waitKey(20);
			
			if(key==32){ //Si se pulsa espacio se guarda la captura actual con formato sal_xxx.png
				cv::Mat imgToSave=outputFrame;
				imgToSave=frame1 & imgToSave; 
				
				outputName = "sal_";
				
				std::stringstream ss;
				ss << cont;
				outputName = outputName + ss.str();
				
				outputName= outputName + ".png";
				cv::imwrite(outputName,imgToSave);
			}
			
			outputFrame=splitOR(outputFrame);
			
			outputVideo << outputFrame;
			
			cv::imshow( "Display window", outputFrame );
			cv::imshow( "Original video", frame1 );
			
			
		}
		cont++;
	}
		
	
   cv::waitKey(0); //Espera hasta que pulses un boton y cierra programa
   
  }catch(std::exception &ex){
    std::cout<<ex.what()<<std::endl;
  }

}
