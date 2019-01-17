#include <opencv2/core/core.hpp> //core routines
#include <opencv2/highgui/highgui.hpp>//imread,imshow,namedWindow,waitKey
#include "opencv2/imgproc/imgproc.hpp"//Para ecualHist
#include <iostream>

#include <vector>

std::vector<float> calculate_histogram(const cv::Mat& img, cv::Mat mask=cv::Mat(), int bins=256, bool normalized=false, bool accumulated=false){

  //bins -> tamanyo del vector
  std::vector<float> vh; //Este vector guardara el histograma y se devolvera al final de la funcion
  vh.resize(bins);
  cv::Mat imagenAux=img;
  
  if( mask.empty() ){
    for(int y=0;y<imagenAux.rows;y++){
      uchar *ptr=imagenAux.ptr<uchar>(y);
      for(int x=0;x<imagenAux.cols;x++,ptr++){
        vh[(int)ptr[0]]++;
      }
     }
  }
  else{
    for(int y=0;y<imagenAux.rows;y++){
      uchar *ptr=imagenAux.ptr<uchar>(y);
      uchar *ptr2=mask.ptr<uchar>(y);
      for(int x=0;x<imagenAux.cols;x++,ptr++,ptr2++){
        if(ptr2[0]==255){
          vh[(int)ptr[0]]++;
        }
        
      }
     }
  }
   
   return vh;

};


std::vector<float> calculate_VAcumulated(std::vector<float> vh, int bins=256){

	
   std::vector<float> vAcumulativo;
   vAcumulativo.resize(bins);
   
   //Calcula El Vector Acumulativo
   vAcumulativo[0]=vh[0];
   for(int i=1;i<vAcumulativo.size();i++){
		vAcumulativo[i]=vh[i]+vAcumulativo[i-1];
   }
   
   return vAcumulativo;

};

std::vector<float> calculate_VEcualizador(int totalPixels, std::vector<float> vAcumulativo, int bins=256){

   
   std::vector<float> vEcualizador;
   vEcualizador.resize(bins);
   
   
   for(int i=0;i<bins;i++){
		//vEcualizador[i]=((vAcumulativo[i]*255)/totalPixels);
		
		//////////////////Para redondear a entero
		vEcualizador[i]=((vAcumulativo[i]*255)/totalPixels);
		int aux=(int)vEcualizador[i];
		if( ( vEcualizador[i] - aux ) > 0.5 ){
			vEcualizador[i]++;
		}
		/////////////////
   }   
   
   return vEcualizador;
      
};

cv::Mat img_ecualizada(cv::Mat img){

  std::vector<float> vh;
  std::vector<float> vAc;
  std::vector<float> vEc;
  int totalPixels= img.rows * img.cols;

  totalPixels= img.rows * img.cols;
	
   
  vh=calculate_histogram(img);
  vAc=calculate_VAcumulated(vh);
  vEc=calculate_VEcualizador(totalPixels,vAc);


  cv::Mat imgAux(img.rows,img.cols,CV_8UC1); //imagen del mismo tamanyo que img, de 1 canal
   

    for(int y=0;y<imgAux.rows;y++){
      uchar *ptr=imgAux.ptr<uchar>(y);
      uchar *ptr2=img.ptr<uchar>(y);
      for(int x=0;x<imgAux.cols;x++,ptr++,ptr2++){
      	ptr[0]=vEc[ptr2[0]];
  	  }
  	  
    }

   return imgAux;

};


/*
cv::Mat img_ecualizada(cv::Mat img, cv::Mat mask=cv::Mat(), int r=0, int bins=256){
  

  if(r==0){
    std::vector<float> histogram;
    std::vector<float> vAc;
    std::vector<float> vEc;
    
    int totalPixels=img.rows * img.cols;
    if( mask.empty() ){
    
      histogram=calculate_histogram(img);
	  vAc=calculate_VAcumulated(histogram);
      vEc=calculate_VEcualizador(totalPixels,vAc);

    }
    else{
      totalPixels=pixelsMask(mask);
      
      histogram=calculate_histogram(img,mask);
	  vAc=calculate_VAcumulated(histogram);
	  vEc=calculate_VEcualizador(totalPixels, vAc);
	}
    
    
      cv::Mat imgAux(img.rows,img.cols,CV_8UC1); //imagen del mismo tamanyo que img, de 1 canal
      for(int y=0;y<imgAux.rows;y++){
        uchar *ptr=imgAux.ptr<uchar>(y);
        uchar *ptr2=img.ptr<uchar>(y);
        for(int x=0;x<imgAux.cols;x++,ptr++,ptr2++){
      	  ptr[0]=vEc[ptr2[0]];
  	    }
  	  
      }
    return imgAux;
  }
  else{
  int ventana=(2*r)+1;
  
  std::vector<float> vh;
  std::vector<float> vAc;
  std::vector<float> vEc;
  int totalPixels= ventana*ventana;
  
  cv::Mat imagenAux(ventana,ventana,CV_8UC1);
  
  cv::Mat imagenFinal(img.rows,img.cols,CV_8UC1);

  for(int y=0; y<imagenFinal.rows; y++){
    uchar *ptr=img.ptr<uchar>(y);
    uchar *ptr2=imagenFinal.ptr<uchar>(y);
    for(int x=0; x<imagenFinal.cols; x++,ptr++,ptr2++){

      
      if( (x+r <= imagenFinal.cols) && (x-r >= 0) ){
      	if( (y+r <= imagenFinal.rows) && (y-r >= 0) ){
      	
      	  cv::Range rowRange(y-r,y+r);
          cv::Range colRange(x-r,x+r);
          imagenAux = img.operator()(rowRange, colRange);
          
          vh.resize(bins);
	      vAc.resize(bins);
	      vEc.resize(bins);
	  
          vh=calculate_histogram(imagenAux);
          vAc=calculate_VAcumulated(vh);
	      vEc=calculate_VEcualizador(totalPixels,vAc);
	      //imagenAux=img_ecualizada(img,vEc);
	      ptr2[0]=vEc[ptr[0]];
      	
      	}
      	else{
          ptr2[0]=ptr[0];
      	}
      }
      else{
      	ptr2[0]=ptr[0];
      }
      
      
      
     } //cierro for x
   } //cierro for y
   return imagenFinal;
  }


};*/






cv::Mat img_ecualizada_radio(cv::Mat img, int r=0, int bins=256){
  
  int ventana=(2*r)+1;
  
  std::vector<float> vh;
  std::vector<float> vAc;
  std::vector<float> vEc;
  int totalPixels= ventana*ventana;
  
  cv::Mat imagenAux(ventana,ventana,CV_8UC1);
  
  cv::Mat imagenFinal(img.rows,img.cols,CV_8UC1);

  for(int y=0; y<imagenFinal.rows; y++){
    uchar *ptr=img.ptr<uchar>(y);
    uchar *ptr2=imagenFinal.ptr<uchar>(y);
    for(int x=0; x<imagenFinal.cols; x++,ptr++,ptr2++){
      /*if( (x+r)>img.cols ){
        ptr2[0]=ptr[0];
      }
      if( (y+r)>img.rows ){
        ptr2[0]=ptr[0];
      }*/
      
      if( (x+r <= imagenFinal.cols) && (x-r >= 0) ){
      	if( (y+r <= imagenFinal.rows) && (y-r >= 0) ){
      	
      	  cv::Range rowRange(y-r,y+r);
          cv::Range colRange(x-r,x+r);
          imagenAux = img.operator()(rowRange, colRange);
          
          vh.resize(bins);
	      vAc.resize(bins);
	      vEc.resize(bins);
	  
          vh=calculate_histogram(imagenAux);
          vAc=calculate_VAcumulated(vh);
	      vEc=calculate_VEcualizador(totalPixels,vAc);
	      //imagenAux=img_ecualizada(img,vEc);
	      ptr2[0]=vEc[ptr[0]];
      	
      	}
      	else{
          ptr2[0]=ptr[0];
      	}
      }
      else{
      	ptr2[0]=ptr[0];
      }
      
      
      
     } //cierro for x
   } //cierro for y
   
   return imagenFinal;

};


int pixelsMask(cv::Mat mask){
	int pixels=0;
	for(int y=0;y<mask.rows;y++){
      uchar *ptr=mask.ptr<uchar>(y);
      for(int x=0;x<mask.cols;x++,ptr++){
        if(ptr[0]==255){
          pixels++;
        }
      }
     }

	return pixels;
};


cv::Mat img_ecualizada_mask(cv::Mat img, cv::Mat mask){

  std::vector<float> vh;
  std::vector<float> vAc;
  std::vector<float> vEc;
  int totalPixels=pixelsMask(mask);
   
  vh=calculate_histogram(img,mask);
  vAc=calculate_VAcumulated(vh);
  vEc=calculate_VEcualizador(totalPixels, vAc);
  
  
  cv::Mat imgAux(img.rows,img.cols,CV_8UC1); //imagen del mismo tamanyo que img, de 1 canal
   

    for(int y=0;y<imgAux.rows;y++){
      uchar *ptr=imgAux.ptr<uchar>(y);
      uchar *ptr2=img.ptr<uchar>(y);
      uchar *ptr3=mask.ptr<uchar>(y);
      for(int x=0;x<imgAux.cols;x++,ptr++,ptr2++,ptr3++){
        if(ptr3[0]==255){ //Si en la mascara detecto pixel blanco
        	ptr[0]=vEc[ptr2[0]];
        }
        else{
      		ptr[0]=ptr2[0];
      	}
  	  }
  	  
    }

   return imgAux;

};

cv::Mat img_ecualizada_mask_radio(cv::Mat img, cv::Mat mask, int r=0, int bins=256){

  int ventana=(2*r)+1;
  
  std::vector<float> vh;
  std::vector<float> vAc;
  std::vector<float> vEc;
  int totalPixels=0;
  
  cv::Mat imagenAux(ventana,ventana,CV_8UC1);
  
  cv::Mat imagenFinal(img.rows,img.cols,CV_8UC1);
  
  for(int y=0; y<imagenFinal.rows; y++){
    uchar *ptr=img.ptr<uchar>(y);
    uchar *ptr2=imagenFinal.ptr<uchar>(y);
    uchar *ptr3=mask.ptr<uchar>(y);
    for(int x=0; x<imagenFinal.cols; x++,ptr++,ptr2++,ptr3++){

      
      if( (x+r <= imagenFinal.cols) && (x-r >= 0) && (ptr3[0]==255) ){
      	if( (y+r <= imagenFinal.rows) && (y-r >= 0) ){ //Estoy dentro de la ventana
      	
      	  cv::Range rowRange(y-r,y+r);
          cv::Range colRange(x-r,x+r);
          imagenAux = img.operator()(rowRange, colRange);
          
          vh.resize(bins);
	      vAc.resize(bins);
	      vEc.resize(bins);
	      
	  
          vh=calculate_histogram(imagenAux);
          vAc=calculate_VAcumulated(vh);
	      vEc=calculate_VEcualizador(vAc.back(),vAc); // vAc.back()==Ultimo elemento del vector vAc

	      ptr2[0]=vEc[ptr[0]];
      	
      	}
      	else{
          ptr2[0]=ptr[0];
      	}
      }
      else{
      	ptr2[0]=ptr[0];
      }
      
      
      
     } //cierro for x
   } //cierro for y
   
   return imagenFinal;


};





int main(int argc,char **argv){
try{

  if(argc<3) {std::cerr<<"USE--> ./practica1.exe [-r <int>] <input img> <output img> [mask]"<<std::endl;return 0;}   
  cv::Mat image; //imagen a procesar
  cv::Mat imageEc; //imagen procesada

  char * outputName; //nombre de la imagen procesada
  
   if(argc==4){ //se usa [mask]
		cv::Mat mask;
   		image=cv::imread(argv[1], 0);//Lee la imagen EN MONOCOLOR
   		if( image.rows==0) {std::cerr<<"Error reading image"<<std::endl;return 0;}
   		
   		mask=cv::imread(argv[3], 0);//Lee la mascara EN MONOCOLOR
   		if( image.rows==0) {std::cerr<<"Error reading mask"<<std::endl;return 0;}
   		
   		outputName=strcat(argv[2],".jpg"); //nombre de la imagen de salida
   		
		imageEc=img_ecualizada_mask(image, mask);
   
   }
   else if(argc==5){ //se usar -r <int>
		int r = atoi(argv[2]);
		//Leo imagen
   		image=cv::imread(argv[3], 0);//Lee la imagen EN MONOCOLOR
   		if( image.rows==0) {std::cerr<<"Error reading image"<<std::endl;return 0;}
   		
   		outputName=strcat(argv[4],".jpg"); //nombre de la imagen de salida
   		

		imageEc=img_ecualizada_radio(image, r); //con ventana
		
   }
   else if(argc==6){ //se usa -r <int> y [mask]
   		int r = atoi(argv[2]);
		//Leo imagen
		cv::Mat mask;
   		image=cv::imread(argv[3], 0);//Lee la imagen EN MONOCOLOR
   		if( image.rows==0) {std::cerr<<"Error reading image"<<std::endl;return 0;}
   		
   		mask=cv::imread(argv[5], 0);//Lee la mascara EN MONOCOLOR
   		if( image.rows==0) {std::cerr<<"Error reading mask"<<std::endl;return 0;}
   		
   		outputName=strcat(argv[4],".jpg"); //nombre de la imagen de salida
   		
		
		imageEc=img_ecualizada_mask_radio(image,mask,r); //con ventana y mask
   		
		
   		
   }
   else if(argc==3){ //se usa solo <input img> <output img>
		//loads the image from file
		image=cv::imread(argv[1], 0);//Lee la imagen EN MONOCOLOR
		if( image.rows==0) {std::cerr<<"Error reading image"<<std::endl;return 0;}
     
		outputName=strcat(argv[2],".jpg"); //nombre de la imagen de salida
		
		imageEc=img_ecualizada(image);
   }

   

   cv::imwrite(outputName,imageEc);

   //creates a window
   cv::namedWindow("Image-Original");
   //displays the image in the window
   cv::imshow("Image-Original",image);
   
   
   //cv::namedWindow("Image-ECU");
   cv::imshow("Image-ECU",imageEc);
   
   cv::waitKey(0); //Espera hasta que pulses un boton y cierra programa
   
  }catch(std::exception &ex){
    std::cout<<ex.what()<<std::endl;
  }

}
