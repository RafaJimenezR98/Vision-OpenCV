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

	//"-r=0.5  | valor-por-defecto |  description"
	//Cuando se usa @, se indica el orden(como un vector)
const cv::String keys =    
    "{@rows        		  |                   | rows table     }"
    "{@cols        		  |                   | cols table     }"
    "{@size        		  |300                | size 3D	   }"
    "{@intrinsics         |logitech.yml       | intrinsics parameters   }"
    "{@videoFile          |tablero_000_000.avi| vid   }"
    "{i                   |                   | virtual img   }"
    ;


bool detectarTablero(std::vector<cv::Point2f> &esquinas, std::vector<cv::Point3f> &esquinasObjeto, cv::Mat &frame, int rows, int cols, int size){
	
	cv::Mat frameAux; //Para guardarlo en blanco y negro
		
		
	//Se pasa el frame a escala de grises
	cvtColor(frame,frameAux,CV_BGR2GRAY);
	
	bool exito=cv::findChessboardCorners(frameAux, cv::Size(rows, cols), esquinas);
	
	if(exito){ //si se han detectado bien las esquinas...
		
		//Refina las esquinas obtenidas
		cv::cornerSubPix(frameAux, esquinas, cv::Size(20, 20), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
			
		//dibuja las lineas de calibracion
		/*cv::drawChessboardCorners(frame, cv::Size(rows, cols), esquinas, true);
		std::cout<<"Esquinas encontradas: "<<esquinas.size()<<"\n";
		imshow( "Display window", frame );*/
		//cv::waitKey(0);

		//Introduce puntos al esquinasObjetoAux
		for(int i = 0; i < cols; i++){
			for(int j = 0; j < rows; j++){
       			esquinasObjeto.push_back(cv::Point3f(j*size, i*size, 0.0f));
			}
		}
		
		return true;
   	}
   	else{
   		return false;
   	}

}




void cameraParameters(cv::Mat &cameraMatrix, cv::Mat &distortion, cv::Mat &rotation, cv::Mat &translation){
	//cv::Mat cameraMatrix, distortion, rotation, translation;
	
	cv::FileStorage fs("logitech.yml", cv::FileStorage::READ);
	
	fs["camera-matrix"] >> cameraMatrix;
	fs["distortion-coefficients"] >> distortion;
	fs["rotation-matrix"] >> rotation;
	fs["translation-vector"] >> translation;
	fs.release();

};


void estimarOrientacionCamara(std::vector<cv::Point2f> &esquinas, std::vector<cv::Point3f> &esquinasObjeto, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &cameraMatrix, cv::Mat &distortion){
	
	cv::solvePnP(cv::Mat(esquinasObjeto), cv::Mat(esquinas), cameraMatrix, distortion, rvec, tvec);

};



void proyectar3D(std::vector<cv::Point3f> &points, std::vector<cv::Point2f> &salida, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &cameraMatrix, cv::Mat &distortion, cv::Mat &frame, int rows, int cols, int size){
	points.push_back(cv::Point3f((rows/2)*size,(cols/2)*size,0));
	points.push_back(cv::Point3f(((rows/2)+1)*size,(cols/2)*size,0));
	points.push_back(cv::Point3f((rows/2)*size,((cols/2)+1)*size,0));
	points.push_back(cv::Point3f((rows/2)*size,(cols/2)*size,-size));

	cv::projectPoints(cv::Mat(points), rvec, tvec, cameraMatrix, distortion, salida);
        
	cv::Scalar blue(255,0,0);
	cv::Scalar green(0,255,0);
	cv::Scalar red(0,0,255);


	cv::line(frame,salida[0],salida[1], blue,2); 
	cv::line(frame,salida[0],salida[2], green,2); 
	cv::line(frame,salida[0],salida[3], red,2);

};

void proyectarImg(std::string imageName, cv::Size &videoSize, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &cameraMatrix, cv::Mat &distortion, cv::Mat &frame, int rows, int cols, int size){
	
	cv::Mat img=cv::imread(imageName);
	std::vector<cv::Point2f> esquinasImg;
	std::vector<cv::Point3f> puntosProyectados;
	
	puntosProyectados.push_back(cv::Point3f(rows*size,cols*size,0));
	puntosProyectados.push_back(cv::Point3f(0,0,0));
	puntosProyectados.push_back(cv::Point3f(rows*size,0,0));
	puntosProyectados.push_back(cv::Point3f(0,cols*size,0));

	std::vector<cv::Point2f> salida;  
	cv::projectPoints(cv::Mat(puntosProyectados), rvec, tvec, cameraMatrix, distortion, salida);

	//Va añadiendo al vector de puntos las esquinas de la img a proyectar
	esquinasImg.push_back(cv::Point2f(0,img.cols));
	esquinasImg.push_back(cv::Point2f(img.rows,0));
	esquinasImg.push_back(cv::Point2f(0,0));
	esquinasImg.push_back(cv::Point2f(img.rows,img.cols));
	
	
	cv::Mat perspectiva= getPerspectiveTransform(esquinasImg, salida); //Para calcular la transformacion entre los pares de puntos
	cv::warpPerspective(img, frame, perspectiva, videoSize); //Aplica la transformación de perspectiva calculada antes al frame
	
	
}


int main(int argc,char **argv){

try{
  //augReal rows cols size intrinsics.yml <input video-file|cam-idx>
  
  if(argc<6) {std::cerr<<"USE--> ./augReal rows cols size intrinsics.yml <input video-file|cam idx>"<<std::endl;return 0;}   
  
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Application name v1.0.0");

	
	int rows = parser.get<int>(0);
	int cols = parser.get<int>(1);
	int size = parser.get<int>(2);
	
	cv::String imageName = parser.get<cv::String>("i");
	
	cv::String intrinsics = parser.get<cv::String>(3);
	cv::String video = parser.get<cv::String>(4);
	//video="tablero_000_000.avi";
	//rows=5;
	//cols=4;
	
	
	cv::VideoCapture capture(video);
	if(!capture.isOpened()){ //Error abriendo video
		std::cout<<"Error abriendo el video"<<std::endl;
		return 0;
	}
	cv::Mat cameraMatrix, distortion, rotation, translation;
	cameraParameters(cameraMatrix, distortion, rotation, translation);
	
	while(capture.grab()){
		cv::Mat frame;
		capture.retrieve(frame); //Guarda en frame el frame
		
		std::vector<cv::Point2f> esquinas;
		std::vector<cv::Point3f> esquinasObjeto;
		bool detectaTablero=detectarTablero(esquinas, esquinasObjeto, frame, rows, cols, size);
		
		if(detectaTablero){ //Si detecta un tablero de ajedrez
			cv::Mat rvec, tvec;
			estimarOrientacionCamara(esquinas, esquinasObjeto, rvec, tvec, cameraMatrix, distortion);
			
			if(imageName.empty()){ //Si no le paso una imagen para virtualizarla
				std::vector<cv::Point3f> points;
				std::vector<cv::Point2f> salida;
				proyectar3D(points, salida, rvec, tvec, cameraMatrix, distortion, frame, rows, cols, size);
				imshow( "Display window", frame );
				//cv::waitKey(0);
			}
			else{ //Si le paso imagen para virtualizar
				cv::Size videoSize= cv::Size( capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT) ); //Para obtener el tamaño del video
				
				proyectarImg(imageName, videoSize, rvec, tvec, cameraMatrix, distortion, frame, rows, cols, size);
				imshow( "Display window", frame );
			
			}
			
		}
		
		if (cv::waitKey(10) >= 0) break; //Para que se muestre como video
	}
	
   
   
   cv::waitKey(0); //Espera hasta que pulses un boton y cierra programa
   
  }catch(std::exception &ex){
    std::cout<<ex.what()<<std::endl;
  }

}
