#!/bin/sh

#part 2b, 2c
#Rs="1"
#maxIts="1"
Rs="20"
maxIts="15"
#ocr dataset
echo "Running ocr datasets, for phi1"
python StructuredPerceptron.py --trainPath=ocr_fold0_sm_train.txt --testPath=ocr_fold0_sm_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=1 > ocr_phi1_R20_maxIt15.txt
echo "Running ocr datasets, for phi2"
python StructuredPerceptron.py --trainPath=ocr_fold0_sm_train.txt --testPath=ocr_fold0_sm_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=2 > ocr_phi2_R20_maxIt15.txt
echo "Running ocr datasets, for phi3"
python StructuredPerceptron.py --trainPath=ocr_fold0_sm_train.txt --testPath=ocr_fold0_sm_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=3 > ocr_phi3_R20_maxIt15.txt
#nettalk dataset
echo "Running net datasets, for phi1"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=1 > net_phi1_R20_maxIt15.txt
echo "Running net datasets, for phi2"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=2 > net_phi2_R20_maxIt15.txt
echo "Running net datasets, for phi3"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=$Rs --eta=0.01 --phi=3 > net_phi3_R20_maxIt15.txt

#part 2d
#maxIts="1"
maxIts="15"
echo "Running net datasets, for R10"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=10 --eta=0.01 --phi=1 > net_phi1_R10_maxIt15.txt
echo "Running net datasets, for R25"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=25 --eta=0.01 --phi=1 > net_phi1_R25_maxIt15.txt
echo "Running net datasets, for R50"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=50 --eta=0.01 --phi=1 > net_phi1_R50_maxIt15.txt
echo "Running net datasets, for R100"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=100 --eta=0.01 --phi=1 > net_phi1_R100_maxIt15.txt
echo "Running net datasets, for R200"
python StructuredPerceptron.py --trainPath=nettalk_stress_train.txt --testPath=nettalk_stress_test.txt --maxIt=$maxIts --R=200 --eta=0.01 --phi=1 > net_phi1_R200_maxIt15.txt
