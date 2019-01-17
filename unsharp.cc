#include <opencv2/core/core.hpp> //core routines
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include "opencv2/imgproc/imgproc.hpp"//Para ecualHist
#include <iostream>

#include <exception>
#include <opencv2/core/utility.hpp>

#include <vector>

/*class CmdLineParser{ //Para usar argumentos por linea de comandos
   int argc;
   char **argv;
public:
  CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}
  bool operator[] ( string param ) {int idx=-1; for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; return ( idx!=-1 ) ; }
  string operator()(string param,string defvalue="-1"){int idx=-1; for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue; else return ( argv[ idx+1] ); }
};*/

	//"-r=0.5  | valor-por-defecto |  description"
	//Cuando se usa @, se indica el orden(como un vector)
const cv::String keys =    
    "{r        		  |0      | range of window      }"
    "{g        		  |1      | gain			        }"
    "{f        		  |0      | method for filter	}"
    "{@image1         |       | image1 for compare   }"
    "{@image2         |out.png| image2 for compare   }"
    ;


int windowSize(float r, cv::Mat img){ //calcula el tamanyo de la ventana
	int ventana;
	
	if( img.cols < img.rows ){ //COMPROBAR PQ A VECES SALE TAMANYO DE VENTANA PAR Y OTRAS IMPAR
		ventana = (int)( (r * (img.cols/2) ) +1); //casteo a int para que nos salgan nº pixels enteros
		if(ventana%2 == 0){
			ventana--; //Para que la ventana tenga tamanyo numero impar
		}
	}
	else{
		ventana = (int)( (r * (img.rows/2) ) +1); //casteo a int para que nos salgan nº pixels enteros
		if(ventana%2 == 0){
			ventana--; //Para que la ventana tenga tamanyo numero impar
		}
	}
	if( ventana < 3 ){ //Por si la r introducida es 0
		ventana = 3; //2*1 + 1
	}
	
	return ventana;


};         

cv::Mat createBoxFilter(int windowSize){ //Crea un filtro Box de radio r>0
	// 1/totalpixels(ancho*alto) * (imagen/matriz de valores todo 1)
	//boxFilter();
	//int ventana= r;
	
	cv::Mat filter (windowSize, windowSize, CV_32FC1);
	filter = 1.0/(windowSize * windowSize);

	return filter;

};

void convolve(const cv::Mat& in, const cv::Mat& filter, cv::Mat& out){
//calcula la convolución digital. La imagen de salida tendrá la mismas dimensiones de la imagen 
//de entrada, rellenando con ceros la zona no procesada. Precondiciones: in.type()==CV_32FC1 && filter.type()==CV_32FC1

	//cv::Mat imgAux=cv::Mat::zeros(in.rows, in.cols, CV_32FC1);
	cv::Mat filterAux=filter;
	
	float sumatorio = 0.0;//Variable que guarda el sumatorio de las multiplicaciones de los pixels
	

	assert(filter.cols%2!=0);
	int ventana = (filter.cols)/2; // == filterAux.rows	

	if( in.type() != CV_32FC1 ){
		std::cout<<"La imagen de entrada tiene que tener el formato --> CV_32FC1"<<std::endl;
	}
	if( filterAux.type() != CV_32FC1 ){
		std::cout<<"El filtro tiene que tener el formato --> CV_32FC1"<<std::endl;
	}
	
	for(int y=ventana; y<in.rows-ventana; y++){
		float *ptr2=out.ptr<float>(y);
		for(int x=ventana; x<in.cols-ventana; x++,ptr2++){
		//std::cout<<"valores imagen original-->"<<ptr[0]<<std::endl;
	
			//if( (x+ventana <= in.cols) && (x-ventana >= 0) && (y+ventana <= in.rows) && (y-ventana >= 0) ){
				                         // x > ventana                                 // y > ventana
				//Coges la imagen original con el tamanyo de ventana y coges la ventana y vas multiplicando el primer pixel de la imagen original
				//y lo multiplicas por el ultimo pixel del filtro y lo vas sumando en una variable aux. cuando termines de multiplicar todos los pixels
				// de la la ventana e img, el pixel central le pones el valor de ese sumatorio(aux)
				//Son 4 bucles for anidados
				//La img de salida tiene que tener ceros en los bordes(donde no llegue los limites de la ventana)
		
		
				//cv::Range rowRange(y-ventana,y+ventana);
				//cv::Range colRange(x-ventana,x+ventana);
				//cv::Mat imgAux = in.operator()(rowRange, colRange);
				//cv::Mat imgAux(in, cv::Rect(x-ventana, y-ventana, filter.rows-1, filter.cols-1));
				cv::Mat imgAux = in(cv::Rect(x-ventana, y-ventana, filter.cols, filter.rows));
				//std::cout<<"x-ventana: "<<x-ventana<<std::endl;
				//std::cout<<"y-ventana: "<<y-ventana<<std::endl;
		
				for(int w=0; w<imgAux.rows; w++){
					float *ptr3=imgAux.ptr<float>(w);
					float *ptr4=filterAux.ptr<float>(w);
					for(int z=0; z<imgAux.cols; z++,ptr3++,ptr4++){
						//std::cout<<"valores pixels filtro-->"<<ptr4[0]<<std::endl;
						//std::cout<<"valores pixels imgAux-->"<<ptr3[0]<<std::endl;
						//sumatorio += ptr3[0] * ptr4[0];
						sumatorio += imgAux.at<float>(w,z) * filter.at<float>(w,z);
							
					}

				}
				//std::cout<<"valor sumatorio-->"<<sumatorio<<std::endl;
				//ptr2[0]=sumatorio;
				out.at<float>(y,x)=sumatorio;
				sumatorio=0.0;
				//std::cout<<"valor ptr2-->"<<ptr2[0]<<std::endl;
		
		
			//}
			//else{
			//	ptr2[0]=0;
			//}
	 
		}
	}
	


};

int main(int argc,char **argv){

try{
  //unsharp [-r <float>] [-g <float>] [-f int] <input img> <output img>
  if(argc<3) {std::cerr<<"USE--> ./unsharp [-r <float>] [-g <float>] [-f int] <input img> <output img>"<<std::endl;return 0;}   
  
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Application name v1.0.0");

	float r = parser.get<float>("r");
	float g = parser.get<float>("g");
	int f = parser.get<int>("f");
	
	if( (r>1)||(r<0) ){
		r=0;
	}
	if( (g>10)||(g<0) ){
		g=1;
	}
	if( (f!=0)||(f!=1) ){
		f=0;
	}
	
	cv::String nameImg1 = parser.get<cv::String>(0);
	cv::String nameImg2 = parser.get<cv::String>(1);
  
  
  cv::Mat imageAux; 
  
  imageAux=cv::imread(nameImg1,0);//Lee la imagen EN MONOCOLOR
  if( imageAux.rows==0) {std::cerr<<"Error reading image"<<std::endl;return 0;}
  cv::Mat image; //imagen a procesar
  imageAux.convertTo(image, CV_32FC1);
  
  cv::Mat imageS = cv::Mat::zeros(image.rows, image.cols, CV_32FC1); //imagen procesada
  
  int ventana=windowSize(r, image);
  
  convolve(image, createBoxFilter(ventana) , imageS);
  
  //operación a aplicar es: O = (g+1)·I -g·IL, siendo I la imagen original, IL la versión paso baja de la misma y g la ganancia del realce.
  
   imageS = ( (g+1)*image ) - ( g*imageS);
   
   cv::imwrite(nameImg2,cv::Mat_<uchar>(imageS));

   //creates a window
   cv::namedWindow("Image-Original");
   //displays the image in the window
   cv::imshow("Image-Original", cv::Mat_<uchar>(image));
   
   
   /*cv::namedWindow("image-Albertito", cv::WINDOW_GUI_NORMAL);
   cv::setWindowProperty("image-Albertito", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
   cv::imshow("image-Albertito", cv::Mat_<uchar>(imageS));
   cv::waitKey(0);
   cv::destroyAllWindows();
	*/
   
   
   cv::imshow("Image-Salida", cv::Mat_<uchar>(imageS));

   
   
   cv::waitKey(0); //Espera hasta que pulses un boton y cierra programa
   
  }catch(std::exception &ex){
    std::cout<<ex.what()<<std::endl;
  }

}
