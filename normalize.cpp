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
	char dirn[] = "../../dataset/neutral";
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
						string line;
  						ifstream myfile (("../../dataset/neutral/" + (string)drnt->d_name + ".ldm").c_str());
						if (myfile.is_open()){
							vector<int> idx;
							while ( getline (myfile,line) ){
								vector<string> v = explode(line,' ');
								idx.push_back(stoi(v[0]));
							}
							myfile.close();
							indexes.push_back(idx);
						}
					}
					pch = strtok (NULL, ".");
				}
			}
		}
        	closedir(dir);
		cout << "Loaded " << clouds.size() << " pointclouds" << endl;
		tFimLocal = clock();
		cout << "Carregamento do diretorio executado em " << ((tFimLocal - tInicioLocal)/CLOCKS_PER_SEC)<< "s" << endl;

		if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("../modelo/modelo.pcd", *modelo) == -1){
			PCL_ERROR ("Couldn't read file a file\n");
			return (-1);
		}

		x[1] = numeric_limits<float>::min();
		y[1] = numeric_limits<float>::min();
		z[1] = numeric_limits<float>::min();
		x[0] = numeric_limits<float>::max();
		y[0] = numeric_limits<float>::max();
		z[0] = numeric_limits<float>::max();
		for(int i = 0; i < modelo->size(); i++){
			PointT p = modelo->points[i];
			if(p.x < x[0])
				x[0] = p.x;
			if(p.y < y[0])
				y[0] = p.y;
			if(p.z < z[0])
				z[0] = p.z;
			if(p.x > x[1])
				x[1] = p.x;
			if(p.y > y[1])
				y[1] = p.y;
			if(p.z > z[1])
				z[1] = p.z;
		}

		PointCloudT::Ptr cloud_in (new PointCloudT);
		PointCloudT::Ptr saida (new PointCloudT);

		PointCloudT::Ptr modelo2 (new PointCloudT);
		string line2;
		ifstream myfile2 ("../modelo/modelo.ldm");
		vector<int> vid;
		if (myfile2.is_open()){
			while ( getline (myfile2,line2) ){
				vector<string> v = explode(line2,' ');
				vid.push_back(stoi(v[0]));
			}
			myfile2.close();
		}
		for(int i = 0; i < modelo->size(); i++){
			if((modelo->points[i].y < modelo->points[vid[4]].y + 10) && (modelo->points[i].y > modelo->points[vid[4]].y - 10))
				modelo2->push_back(modelo->points[i]);
		}
		cout << "Modelo carregado com sucesso!" << endl;
		cout << indexes.size() << " | " << nomes.size() << endl;

		for(int i = 0; i < clouds.size(); i++){
			float lx[2], ly[2], lz[2];
			PointCloudT::Ptr input (new PointCloudT);
			tInicioLocal = clock();
			cloud_in->points.clear();
			saida->points.clear();
			int radius = 10;
			for(int j = 0; j < clouds[i]->points.size(); j++){
				double marco = clouds[i]->points[indexes[i][4]].y;
				if((clouds[i]->points[j].y < marco + radius) && (clouds[i]->points[j].y > marco - radius))
					cloud_in->push_back(clouds[i]->points[j]);
			}
			cout << clouds[i]->size() << ", " << cloud_in->size() << " | " << modelo->size() << ", " << modelo2->size() << endl;

			pcl::IterativeClosestPoint<PointT, PointT> icp;
			icp.setInputSource(cloud_in);
			icp.setInputTarget(modelo2);
			icp.setMaximumIterations (20);
			pcl::PointCloud<PointT> r;
			icp.align(r);
			r.points.clear();

			Eigen::Matrix4f transform = icp.getFinalTransformation();
			lx[1] = numeric_limits<float>::min();
			ly[1] = numeric_limits<float>::min();
			lz[1] = numeric_limits<float>::min();
			lx[0] = numeric_limits<float>::max();
			ly[0] = numeric_limits<float>::max();
			lz[0] = numeric_limits<float>::max();
			for(int j = 0; j < clouds[i]->points.size(); j++){
				PointT p = clouds[i]->points[j];
				Eigen::Vector4f v(p.x,p.y,p.z,1);
				Eigen::Vector4f w = transform * v;
				p.x = w[0]/w[3];
				p.y = w[1]/w[3];
				p.z = w[2]/w[3];
				if(p.x < lx[0])
					lx[0] = p.x;
				if(p.y < ly[0])
					ly[0] = p.y;
				if(p.z < lz[0])
					lz[0] = p.z;
				if(p.x > lx[1])
					lx[1] = p.x;
				if(p.y > ly[1])
					ly[1] = p.y;
				if(p.z > lz[1])
					lz[1] = p.z;
				r.points.push_back(p);
			}

			pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
			PointCloudT::Ptr alinhada (new PointCloudT);
			alinhada->resize(r.size());
			for(int k = 0; k < r.size(); k++){
				PointT p = r.points[k];
				p.x = ((x[1] - x[0]) * (p.x - lx[0]))/(lx[1] - lx[0]) + x[0];
				p.y = ((y[1] - y[0]) * (p.y - ly[0]))/(ly[1] - ly[0]) + y[0];
				p.z = ((z[1] - z[0]) * (p.z - lz[0]))/(lz[1] - lz[0]) + z[0];
				alinhada->points[k] = p;
			}

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
					/*p.r = 255;
					p.g = 0;
					p.b = 0;*/
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
			pathfim << "../../base_alinhada/neutral/" << nomes[i] << ".pcd";
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