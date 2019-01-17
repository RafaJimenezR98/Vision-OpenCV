#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <tclap/CmdLine.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "common_code.hpp"

#define IMG_WIDTH 300

int
main(int argc, char * argv[])
{
	TCLAP::CmdLine cmd("Image to clasify with BoVW", ' ', "0.0");

    TCLAP::ValueArg<std::string> imageTest("i", "imageTest", "Image to clasify", false,"test.jpg", "std::string");
	cmd.add(imageTest);
	TCLAP::ValueArg<std::string> clasifier("x", "classifier", "Clasifier", false,"02_SURF_SVM_classifier.yml", "std::string");
	cmd.add(clasifier);
	TCLAP::ValueArg<std::string> dictName("d", "dict", "Dict", false,"02_SURF_SVM_dictionary.yml", "std::string");
	cmd.add(dictName);
	TCLAP::ValueArg<std::string> basenameArg("b", "basename", "basename for the dataset.", false, "./data", "pathname");
	cmd.add(basenameArg);
	TCLAP::ValueArg<std::string> configFile("c", "config_file", "configuration file for the dataset.", false, "02_ObjectCategories_conf.txt", "pathname");
	cmd.add(configFile);
	TCLAP::ValueArg<int> ndesc("n", "ndesc", "[SIFT] Number of descriptors per image. Value 0 means extract all. Default 0.", false, 0, "int");
	cmd.add(ndesc);
	//TCLAP::ValueArg<int> keywords("w", "keywords", "[KMEANS] Number of keywords generated. Default 200.", false, 200, "int");
	//cmd.add(keywords);
	TCLAP::ValueArg<int> knn("", "knn", "[KNN] Number k neighbours used to classify. Default 50.", false, 50, "int");
	cmd.add(knn);
	
	//if sift_surf == 1 --> SIFT
	//if sift_surf == 0 --> SURF
	
	//train_bovw -t 0 --> SURF
	TCLAP::ValueArg<int> sift_surf("t", "sift_surf", "Use SIFT or SURF Default 0.(SURF)", false, 0, "int");
	cmd.add(sift_surf);
	
	//if svm_kernel_type == 0 --> DON'T DO SVM
	//if svm_kernel_type == 1 --> lineal kernel
	//if svm_kernel_type == 2 --> polynomial kernel
	//if svm_kernel_type == 3 --> RBF kernel
	TCLAP::ValueArg<int> svm_kernel_type("s", "svm_kernel_type", "Type of kernel for SVM", false, 1, "int");
	cmd.add(svm_kernel_type);
	
	TCLAP::ValueArg<std::string> vid("v", "vid", "Video for test", false, "", "pathname");
	cmd.add(vid);
	
	//if webcam == 1 -> Use real time video(webcam)
	//if webcam == 0 -> DON'T use real time video(webcam)
	//TCLAP::ValueArg<int> webcam("", "webcam", "Video for test", false, 0, "int");
	//cmd.add(webcam);
	

	cmd.parse(argc, argv);

	//cv::Ptr<cv::ml::KNearest> dict = cv::Algorithm::load<cv::ml::KNearest>(dict.getValue());
	
	//std::cout<<configFile.getValue()<<"\n";
	
	cv::FileStorage dictFile;
    dictFile.open(dictName.getValue(), cv::FileStorage::READ);
    int keywords;
    dictFile["keywords"]>>keywords;
    cv::Ptr<cv::ml::KNearest> dict = cv::Algorithm::read<cv::ml::KNearest>(dictFile.root());
    dictFile.release();
	
	cv::Ptr<cv::ml::StatModel> clasificador;
	
	if( svm_kernel_type.getValue() == 0 ){ // don't do SVM
		cv::Ptr<cv::ml::KNearest> knnClassifier = cv::Algorithm::load<cv::ml::KNearest>(clasifier.getValue());
		clasificador = knnClassifier;
	}
	else{
	
		cv::FileStorage classifierFile;
		classifierFile.open(clasifier.getValue(), cv::FileStorage::READ);
		double C;
		classifierFile["C"]>>C;
		classifierFile.release();
		
		cv::Ptr<cv::ml::SVM> svmClassifier = cv::Algorithm::load<cv::ml::SVM>(clasifier.getValue());
		
		if( svm_kernel_type.getValue() == 1){
            svmClassifier->setKernel( cv::ml::SVM::LINEAR );
        }
		else if( svm_kernel_type.getValue() == 2){
			svmClassifier->setKernel(cv::ml::SVM::POLY);
			svmClassifier->setDegree(1);
		}
		else if( svm_kernel_type.getValue() == 3){
			svmClassifier->setKernel(cv::ml::SVM::RBF);
		}
		else{ //any other different value do linear
			svmClassifier->setKernel( cv::ml::SVM::LINEAR );
		}
            	
		svmClassifier->setC(C); // C param for SVM
            	
		svmClassifier->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
		svmClassifier->setType(cv::ml::SVM::C_SVC);
		
		clasificador = svmClassifier;
	}
	
	if( vid.getValue()!="" ){ //if we use video...
		
		std::string dataset_desc_file = basenameArg.getValue() + "/" + configFile.getValue();
		
		std::vector<std::string> categories;
		std::vector<int> samples_per_cat;

		load_dataset_information(dataset_desc_file, categories, samples_per_cat);
		
		cv::Mat vidFrame;
		cv::VideoCapture video(vid.getValue());
		
		//std::string aux;
		
		while(video.read(vidFrame)){
		
			cv::Mat img;

			cv::cvtColor(vidFrame,img,cv::COLOR_RGB2GRAY); //Imagen de prueba para testear en gris

			cv::resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));                  

			cv::Mat desc;
			if( sift_surf.getValue() == 1 ){
				desc = extractSIFTDescriptors(img,  ndesc.getValue());
			}

			else{
				desc = extractSURFDescriptors(img, ndesc.getValue());
			}
     

			cv::Mat predicted_labels;
        	cv::Mat bovw = compute_bovw(dict,keywords,desc);
       		clasificador->predict(bovw,predicted_labels);
       		//aux=categories[predicted_labels.at<float>(0,0)];
        	//if(aux!=categories[predicted_labels.at<float>(0,0)]){
			//int a=vidFrame.cols-50;
        	//putText(vidFrame,categories[predicted_labels.at<float>(0,0)],cv::Point(10,a), cv::FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2);
        	putText(vidFrame,categories[predicted_labels.at<float>(0,0)],cv::Point(10,vidFrame.cols), cv::FONT_HERSHEY_PLAIN, 4,(255,255,255),2);
        	std::cerr << "Categoria Detectada-->" << categories[predicted_labels.at<float>(0,0)] << "\n";
        	//}
        	
        	cv::imshow("CATEGORIZED VIDEO", vidFrame);
        	if (cv::waitKey(5) >= 0){
				break;
			}
			
		}
		
	}
	else{ //we use image
		
		

		std::string dataset_desc_file = basenameArg.getValue() + "/" + configFile.getValue();
	
		cv::Mat img = cv::imread(imageTest.getValue(), cv::IMREAD_GRAYSCALE); //Imagen de prueba para testear en gris
	
	
	
		//cv::Size size(300,300*img.rows/img.cols);
		cv::resize(img, img, cv::Size(IMG_WIDTH, round(IMG_WIDTH*img.rows / img.cols)));
    	//cv::resize(img,img, size);

		cv::Mat desc;
		//desc.convertTo(descs,CV_32F);
		if(sift_surf.getValue() == 1){
			desc = extractSIFTDescriptors(img, ndesc.getValue());
		}
		else{
			desc = extractSURFDescriptors(img, ndesc.getValue());
		}
	
	
		//cv::Mat bovw = compute_bovw(dict, keywords.getValue(), desc, false);
		cv::Mat bovw = compute_bovw(dict, keywords, desc, false);

		//cv::Mat predicted_labels;

    	std::clog << "\tThere are " << bovw.rows << " test images." << std::endl;

    	//Classify the test samples.
    	std::clog << "\tClassifing test images." << std::endl;
    	cv::Mat predicted_labels;
		//clasificador->findNearest(bovw, knn.getValue(), predicted_labels); //relleno la img predict_labels
		clasificador->predict(bovw, predicted_labels);

		std::vector<std::string> categories;
		std::vector<int> samples_per_cat;

		load_dataset_information(dataset_desc_file, categories, samples_per_cat);

		std::cout << "Total categorias: " << std::endl;
		for (unsigned int i=0; i<categories.size(); i++){
			std::cout << "   Categoría[" << i << "] = " << categories[i] << std::endl;
		}

    	std::cerr << "Categoria-->" << categories[predicted_labels.at<float>(0)] << "\n";
    	//std::cerr<<"Categoria-->"<<predicted_labels.at<float>(0,0)<<"\n";
    
    }

}
