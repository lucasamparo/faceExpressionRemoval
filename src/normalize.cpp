#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <time.h>
#include <cmath>
#include <limits>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace std;

float magnitude(PointT p){
	return sqrt(pow(p.x,2) + pow(p.y,2) + pow(p.z,2));
}

const vector<string> explode(const string& s, const char& c){
	string buff{""};
	vector<string> v;
	
	for(auto n:s){
		if(n != c)
			buff+=n;
		else
			if(n == c && buff != ""){
				v.push_back(buff);
				buff = "";
			}
	}
	if(buff != "")
		v.push_back(buff);
	
	return v;
}

int main(){
	//Controle de tempo
	clock_t tInicioTotal, tFimTotal, tInicioLocal, tFimLocal;
	//Multiple Cloud
	tInicioTotal = clock();
	vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr > clouds;
	vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr > alinhadas;
	PointCloudT::Ptr cloud_icp (new PointCloudT);
	PointCloudT::Ptr modelo (new PointCloudT);
	char dirn[] = argv[1];
    	DIR *dir = NULL;
    	struct dirent *drnt = NULL;
	char * pch, * pch2;
	int linha = 0, pWidth = 0;
	vector<string> nomes;
	float x[2], y[2], z[2];

	//Read random index
	vector<vector<int> > indexes;

        dir=opendir(dirn);
	if(dir){
		tInicioLocal = clock();
		while(drnt = readdir(dir)){
			if((drnt->d_name[0] != '.')&&(drnt->d_name[0] != 'm')){
				pch = strtok (drnt->d_name,".");
				while (pch != NULL){
					if(strcmp(pch,"pcd") == 0){
						pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
						stringstream path;
						path << dirn << "/" << drnt->d_name << ".pcd";
						if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (path.str().c_str(), *cloud) == -1){
							PCL_ERROR ("Couldn't read file a file\n");
							return (-1);
						}
						cout << drnt->d_name << " lido com sucesso" << endl;
						clouds.push_back(cloud);
						nomes.push_back((string) drnt->d_name);
					}
					pch = strtok (NULL, ".");
				}
			}
		}
        	closedir(dir);
		cout << "Loaded " << clouds.size() << " pointclouds" << endl;
		tFimLocal = clock();
		cout << "Carregamento do diretorio executado em " << ((tFimLocal - tInicioLocal)/CLOCKS_PER_SEC)<< "s" << endl;

		if (pcl::io::loadPCDFile<PointT> ("../utilities/model.pcd", *modelo) == -1){
			PCL_ERROR ("Couldn't read file a file\n");
			return (-1);
		}

		PointCloudT::Ptr cloud_in (new PointCloudT);
		PointCloudT::Ptr saida (new PointCloudT);
		
		cout << "Modelo carregado com sucesso!" << endl;
		cout << indexes.size() << " | " << nomes.size() << endl;

		for(int i = 0; i < clouds.size(); i++){
			float lx[2], ly[2], lz[2];
			PointCloudT::Ptr input (new PointCloudT);
			tInicioLocal = clock();
			cloud_in->points.clear();
			saida->points.clear();
			cloud_in = clouds[i];
			
			pcl::IterativeClosestPoint<PointT, PointT> icp;
			icp.setInputSource(cloud_in);
			icp.setInputTarget(modelo);
			icp.setMaximumIterations(20);
			pcl::PointCloud<PointT> r;
			icp.align(r);
			r.points.clear();

			Eigen::Matrix4f transform = icp.getFinalTransformation();
			
			pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
			PointCloudT::Ptr alinhada (new PointCloudT);
			kdtree.setInputCloud(modelo);
			int K = 1;
			vector<int> busca(K);
			vector<float> euler(K);
			cloud_icp->points.clear();
			cloud_icp->resize(modelo->size());

			for(int j = 0; j < alinhada->size(); j++){
				if ( kdtree.nearestKSearch (alinhada->points[j], K, busca, euler) > 0 ){
					int a = 0;
					PointT p = alinhada->points[j];
					cloud_icp->points[busca[a]] = p;
				}
			}

			pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree2;
			int K2 = 5;
			vector<int> busca2(K2);
			vector<float> euler2(K2);
			kdtree2.setInputCloud(alinhada);

			for(int j = 0; j < cloud_icp->size(); j++){
				if(magnitude(cloud_icp->points[j]) == 0){
					if ( kdtree2.nearestKSearch (modelo->points[j], K2, busca2, euler2) > 0 ){
						PointT p;
						p.x = 0;
						p.y = 0;
						p.z = 0;
						for(int a = 0; a < busca2.size(); a++){
							p.x = p.x + alinhada->points[busca2[a]].x;
							p.y = p.y + alinhada->points[busca2[a]].y;
							p.z = p.z + alinhada->points[busca2[a]].z;
						}
						p.x = p.x / busca2.size();
						p.y = p.y / busca2.size();
						p.z = p.z / busca2.size();
						/*p.r = 255;
						p.g = 0;
						p.b = 0;*/
						cloud_icp->points[j] = p;
					}
				}
			}

			stringstream pathfim;
			pathfim << argv[2] << nomes[i] << ".pcd";
			pcl::io::savePCDFile(pathfim.str(),*cloud_icp);
			tFimLocal = clock();
 			cout << i << "# - " << nomes[i] << " alinhado ("<< r.size() << "-" << cloud_icp->size() << ") salva com sucesso em " << ((tFimLocal - tInicioLocal) / (CLOCKS_PER_SEC / 1000)) << "ms" << endl;
		}
        } else{
                printf("Não foi possível abrir essa pasta '%s'\n", dirn);
        }
	tFimTotal = clock();
	cout << "Processamento total executado em " << ((tFimTotal - tInicioTotal)/CLOCKS_PER_SEC) << "s" << endl;

	return 0;
}
