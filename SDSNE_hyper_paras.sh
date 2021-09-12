# !/bin/bash

for ds in bbcsport_2view msrcv1 100leaves 3sources scene-15;
do
	if [ "$ds" = "bbcsport_2view" ]
	then
		python execute.py \
								--dataset $ds \
								--mu 0.769\
								--lr 1e-4 \
								--knn 14 \
								--cuda 1 \
								--alpha 0.5 \
								--sc 1
		python execute.py \
								--dataset $ds \
								--mu 0.1\
								--lr 1e-4 \
								--knn 59 \
								--cuda 1 \
								--alpha 0.5 \
								--sc 0
	else
		if [ "$ds" = "msrcv1" ]
		then
			python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-4 \
								--knn 16 \
								--cuda 1 \
								--alpha 0.769 \
								--sc 1
			python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-5 \
								--knn 15 \
								--cuda 1 \
								--alpha 0 \
								--sc 0
		else
			if [ "$ds" = "100leaves" ]
			then
				python execute.py \
								--dataset $ds \
								--mu 0.3 \
								--lr 1e-4 \
								--knn 12 \
								--cuda 1 \
								--alpha 0.3 \
								--sc 1
				python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-4 \
								--knn 14 \
								--cuda 1 \
								--alpha 0.2 \
								--sc 0
			else
				if [ "$ds" = "3sources" ]
				then
					python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-4 \
								--knn 16 \
								--cuda 1 \
								--alpha 0.769 \
								--sc 1
					python execute.py \
								--dataset $ds \
								--mu 0.1 \
								--lr 1e-4 \
								--knn 40 \
								--cuda 1 \
								--alpha 0.5 \
								--sc 0
				else
					if [ "$ds" = "scene-15" ]
					then
						python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-5 \
								--knn 7 \
								--cuda 1 \
								--alpha 0.769 \
								--sc 1
						python execute.py \
								--dataset $ds \
								--mu 0.769 \
								--lr 1e-5 \
								--knn 12 \
								--cuda 1 \
								--alpha 3.0 \
								--sc 0
					fi
				fi
			fi
		fi
	fi
done